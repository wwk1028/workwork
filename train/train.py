"""Training script for the Seq2Seq machine translation model.

Supports:
  - Ctrl+C safe interrupt: saves checkpoint before exiting
  - Resume from checkpoint: python train.py --resume checkpoints/latest.pt
"""

import os
import math
import sys
import signal
import argparse
import torch
import torch.optim as optim

# Ensure HF uses the mirror (China network)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from train.config import Config
from train.seq2seq import Seq2Seq, Seq2SeqLoss
from train.dataset import create_dataloaders

# Global flag for graceful shutdown
_interrupted = False


def _signal_handler(signum, frame):
    global _interrupted
    print("\n[!] Interrupted. Will save checkpoint after this epoch ...")
    _interrupted = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def warmup_cosine_schedule(warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


def save_checkpoint(model, optimizer, scheduler, scaler, config,
                    epoch, global_step, best_loss, path,
                    completed_epoch: bool = False, data_step: int = 0):
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "completed_epoch": completed_epoch,
        "data_step": data_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
        "config": config,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)


def train_epoch(model, dataloader, criterion, optimizer, scheduler, scaler,
                config, epoch, global_step, best_loss, step_offset: int = 0):
    model.train()
    total_loss = 0.0
    accum_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    mid_path = os.path.join(config.save_dir, "mid_epoch.pt")

    for step, (src_ids, src_mask, tgt_ids, decoder_input, tgt_pad_mask) in enumerate(pbar):
        if _interrupted:
            save_checkpoint(model, optimizer, scheduler, scaler, config,
                            epoch, global_step, best_loss, mid_path,
                            completed_epoch=False, data_step=step_offset + step)
            print(f"\n  [!] Interrupted at data #{step_offset + step}. Saved to {mid_path}")
            break

        src_ids = src_ids.to(config.device)
        src_mask = src_mask.to(config.device)
        tgt_ids = tgt_ids.to(config.device)
        decoder_input = decoder_input.to(config.device)
        tgt_pad_mask = tgt_pad_mask.to(config.device)

        with torch.autocast(device_type="cuda", enabled=config.use_amp):
            logits = model(src_ids, src_mask, decoder_input, tgt_pad_mask)
            loss = criterion(logits, tgt_ids)
            loss = loss / config.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_loss += loss.item()

        if (step + 1) % config.grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            global_step += 1
            total_loss += accum_loss
            pbar.set_postfix(loss=accum_loss * config.grad_accum_steps,
                             lr=scheduler.get_last_lr()[0])
            accum_loss = 0.0

            if global_step % config.log_interval == 0:
                avg_loss = total_loss / global_step
                pbar.set_description(f"Epoch {epoch} | avg loss {avg_loss:.4f}")

            # Mid-epoch checkpoint save (completed_epoch=False, save data position)
            if config.save_steps > 0 and global_step % config.save_steps == 0:
                save_checkpoint(model, optimizer, scheduler, scaler, config,
                                epoch, global_step, best_loss, mid_path,
                                completed_epoch=False, data_step=step_offset + step + 1)
                pbar.write(f"  [checkpoint] data #{step_offset + step + 1} step #{global_step}")

    n_updates = max(1, global_step)
    return total_loss / n_updates, global_step


@torch.no_grad()
def evaluate(model, dataloader, criterion, config):
    model.eval()
    total_loss = 0.0
    amp_ctx = torch.autocast(device_type="cuda", enabled=config.use_amp)

    for src_ids, src_mask, tgt_ids, decoder_input, tgt_pad_mask in tqdm(dataloader, desc="Eval"):
        src_ids = src_ids.to(config.device)
        src_mask = src_mask.to(config.device)
        tgt_ids = tgt_ids.to(config.device)
        decoder_input = decoder_input.to(config.device)
        tgt_pad_mask = tgt_pad_mask.to(config.device)

        with amp_ctx:
            logits = model(src_ids, src_mask, decoder_input, tgt_pad_mask)
            total_loss += criterion(logits, tgt_ids).item()

    return total_loss / len(dataloader)


def main():
    config = Config()
    os.makedirs(config.save_dir, exist_ok=True)

    # --- Resume logic ---
    start_epoch = 1
    global_step = 0
    best_loss = float("inf")
    skip_samples = 0
    data_step = 0

    if config.resume_from and os.path.exists(config.resume_from):
        print(f"[0/4] Resuming from {config.resume_from} ...")
        ckpt = torch.load(config.resume_from, map_location=config.device,
                          weights_only=False)
        saved_config = ckpt.get("config", config)
        # Preserve CLI / manual overrides from current config
        saved_config.resume_from = config.resume_from
        saved_config.device = config.device
        saved_config.resume_skip_steps = config.resume_skip_steps
        config = saved_config

        start_epoch = ckpt["epoch"] + ckpt.get("completed_epoch", True)
        global_step = ckpt.get("global_step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        if not ckpt.get("completed_epoch", True):
            data_step = config.resume_skip_steps or ckpt.get("data_step", 0) or 0
            skip_samples = data_step * config.batch_size
            print(f"  Mid-epoch checkpoint: resume epoch {start_epoch} from data #{data_step}")
        print(f"  Resumed at epoch {start_epoch}, step {global_step}, best_loss {best_loss:.4f}")

        print("[1/4] Building dataloaders ...")
        train_loader, valid_loader = create_dataloaders(config, skip_samples=skip_samples)

        print("[2/4] Building model ...")
        model = Seq2Seq(config).to(config.device)
        model.load_state_dict(ckpt["model_state_dict"])

        criterion = Seq2SeqLoss(pad_idx=config.pad_idx,
                                label_smoothing=config.label_smoothing)
        optimizer = optim.Adam(model.parameters(), lr=config.lr,
                               betas=config.betas, eps=config.eps)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        total_steps = (len(train_loader) // config.grad_accum_steps) * config.max_epochs
        scheduler = LambdaLR(
            optimizer,
            warmup_cosine_schedule(config.warmup_steps, total_steps)
        )
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp) if config.device == "cuda" else None
        if "scaler_state_dict" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
    else:
        print("[1/4] Building dataloaders ...")
        train_loader, valid_loader = create_dataloaders(config)

        print("[2/4] Building model ...")
        model = Seq2Seq(config).to(config.device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr,
                               betas=config.betas, eps=config.eps)
        criterion = Seq2SeqLoss(pad_idx=config.pad_idx,
                                label_smoothing=config.label_smoothing)
        total_steps = (len(train_loader) // config.grad_accum_steps) * config.max_epochs
        scheduler = LambdaLR(
            optimizer,
            warmup_cosine_schedule(config.warmup_steps, total_steps)
        )
        scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp) if config.device == "cuda" else None

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}  Trainable: {trainable_params:,}")

    if start_epoch > config.max_epochs:
        print("Training already completed (start_epoch > max_epochs). Exiting.")
        return

    print("[3/4] Training ...")
    try:
        for epoch in range(start_epoch, config.max_epochs + 1):
            train_loss, global_step = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, scaler,
                config, epoch, global_step, best_loss,
                step_offset=(skip_samples // config.batch_size)
            )
            skip_samples = 0  # only the first resumed epoch needs the offset

            if _interrupted:
                # Checkpoint already saved inside train_epoch
                sys.exit(0)

            valid_loss = evaluate(model, valid_loader, criterion, config)

            print(f"  Epoch {epoch:2d}  |  train loss {train_loss:.4f}  |  valid loss {valid_loss:.4f}")

            if valid_loss < best_loss:
                best_loss = valid_loss
                save_checkpoint(model, optimizer, scheduler, scaler, config,
                                epoch, global_step, best_loss,
                                os.path.join(config.save_dir, "best_model.pt"),
                                completed_epoch=True)
                print(f"  --> saved best model (valid loss {best_loss:.4f})")

            if epoch % config.save_interval == 0:
                save_checkpoint(model, optimizer, scheduler, scaler, config,
                                epoch, global_step, best_loss,
                                os.path.join(config.save_dir, f"checkpoint_epoch{epoch}.pt"),
                                completed_epoch=True)

            # Always save latest for resume
            save_checkpoint(model, optimizer, scheduler, scaler, config,
                            epoch, global_step, best_loss,
                            os.path.join(config.save_dir, "latest_checkpoint.pt"),
                            completed_epoch=True)

    except KeyboardInterrupt:
        path = os.path.join(config.save_dir, "latest_checkpoint.pt")
        save_checkpoint(model, optimizer, scheduler, scaler, config,
                        epoch, global_step, best_loss, path,
                        completed_epoch=True)
        print("=" * 60)
        print(f"\n[!] Saved checkpoint at epoch {epoch} to {path}")
        print("=" * 60)
        sys.exit(0)

    print(f"[4/4] Done. Best valid loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()

"""Inference script using the trained Seq2Seq model with Beam Search."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import argparse

from train.config import Config
from train.seq2seq import Seq2Seq
from train.bert_embedding import BERTEmbedding
from train.beam_search import BeamSearch


def load_model(checkpoint_path: str, device: str = "cuda"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    # Backward-compat: ensure new fields exist on old checkpoints
    if not hasattr(config, "lang_tags") or config.lang_tags is None:
        config.lang_tags = {"zh": "<2zh>", "en": "<2en>"}
    if not hasattr(config, "lang_pairs") or config.lang_pairs is None:
        config.lang_pairs = [("en", "zh"), ("zh", "en")]
    model = Seq2Seq(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def translate(model, config, sentences: list, direction: str = None):
    """Translate sentences in the given direction.

    Args:
        direction: target language code, e.g. "zh" or "en".
                   The corresponding tag (e.g. "<2zh>") is prepended to the source.
                   If None, sentences must already include the tag.
    """
    tokenizer = BERTEmbedding(
        config.bert_model_name, config.d_model,
        local_path=config.bert_local_path
    ).tokenizer
    beam = BeamSearch(
        beam_size=config.beam_size,
        max_len=config.max_decode_len,
        length_penalty=config.length_penalty,
        sos_idx=tokenizer.cls_token_id or 101,
        eos_idx=tokenizer.sep_token_id or 102,
        pad_idx=tokenizer.pad_token_id or 0,
    )

    results = []

    for sentence in sentences:
        # Prepend language tag if direction is specified
        if direction:
            tag = config.lang_tags.get(direction, f"<2{direction}>")
            src_text = f"{tag} {sentence}"
        else:
            src_text = sentence

        enc = tokenizer(src_text, max_length=config.max_len, truncation=True,
                        padding="max_length", return_tensors="pt")
        src_ids = enc["input_ids"].to(config.device)
        src_mask = enc["attention_mask"].to(config.device)

        hypotheses = beam.search(model, src_ids, src_mask)

        # Decode top hypothesis
        best_tokens = hypotheses[0].tokens
        best_tokens = [t for t in best_tokens
                       if t not in (beam.sos_idx, beam.eos_idx, beam.pad_idx)]
        translation = tokenizer.decode(best_tokens, skip_special_tokens=True)
        results.append({
            "source": sentence,
            "translation": translation.strip(),
            "score": hypotheses[0].score,
            "n_best": [
                {"tokens": h.tokens, "score": h.score}
                for h in hypotheses
            ]
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Translate with Seq2Seq + Beam Search")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--input", type=str, help="Input text")
    parser.add_argument("--input-file", type=str, help="Input file (one sentence per line)")
    parser.add_argument("--direction", type=str, default=None,
                        help="Target language: 'zh' or 'en'. Prepends <2zh>/<2en> tag.")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output", type=str, help="Output file")
    args = parser.parse_args()

    model, config = load_model(args.model, args.device)
    if args.beam_size:
        config.beam_size = args.beam_size

    if args.input:
        sentences = [args.input]
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
    else:
        print("Enter text to translate (empty line to finish):")
        sentences = []
        while True:
            line = input()
            if not line:
                break
            sentences.append(line)

    results = translate(model, config, sentences, direction=args.direction)

    for r in results:
        print(f"Source:      {r['source']}")
        print(f"Translation: {r['translation']}")
        print(f"Score:       {r['score']:.4f}")
        print("-" * 60)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(r["translation"] + "\n")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

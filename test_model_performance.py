"""
Comprehensive model performance & inference benchmark test.

Tests:
  1. Model Performance Metrics:
     - BLEU-1/2/3/4 scores (sacreBLEU) for zh→en and en→zh
     - Case-insensitive BLEU for comparison
  2. System Inference Performance:
     - Single-sentence latency (various beam sizes)
     - Batch throughput (samples/sec)
     - GPU memory usage (peak & idle)
     - GPU utilization
     - CPU utilization

Usage:
    python test_model_performance.py                    # Full test (BLEU on 2000 sentences)
    python test_model_performance.py --quick            # Quick test (BLEU on 200 sentences)
    python test_model_performance.py --skip-bleu        # Skip BLEU, only inference benchmark
    python test_model_performance.py --output report.json  # Save results to JSON
"""

import os
import sys
import time
import json
import argparse
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import psutil
import numpy as np

# ── Optional: sacreBLEU for standard BLEU ──
try:
    from sacrebleu.metrics import BLEU as SacreBLEU
    _HAS_SACREBLEU = True
except ImportError:
    _HAS_SACREBLEU = False
    print("[WARN] sacrebleu not installed. Will use NLTK BLEU as fallback.")
    print("       Install: pip install sacrebleu")

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False
    print("[WARN] nltk not installed. BLEU computation will be limited.")

# ── GPU monitoring ──
try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False
    print("[WARN] pynvml not available. GPU metrics will be skipped.")


# ============================================================
# Config
# ============================================================

@dataclass
class BenchmarkConfig:
    opus_zh2en_dir: str = "opus_finetuned_full"
    opus_en2zh_dir: str = "opus_en2zh"
    valid_en_path: str = "data/valid.en"
    valid_zh_path: str = "data/valid.zh"
    bleu_sample_size: int = 2000    # sentences for BLEU test
    batch_sizes: tuple = (1, 8, 32)
    beam_sizes: tuple = (1, 4, 5, 6, 10)
    warmup_rounds: int = 3
    benchmark_rounds: int = 10
    max_length: int = 512


cfg = BenchmarkConfig()


# ============================================================
# GPU / CPU Monitor
# ============================================================

class GPUMonitor:
    """Background GPU stat sampler using pynvml."""

    def __init__(self, device_idx: int = 0, interval: float = 0.1):
        self.device_idx = device_idx
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.samples: List[Dict] = []

    def start(self):
        if not _HAS_NVML:
            return
        self._stop.clear()
        self.samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> List[Dict]:
        if not _HAS_NVML:
            return []
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.samples

    def _run(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_idx)
        while not self._stop.is_set():
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.samples.append({
                    "ts": time.perf_counter(),
                    "gpu_util_pct": util.gpu,
                    "mem_used_mb": mem.used / (1024 ** 2),
                    "mem_total_mb": mem.total / (1024 ** 2),
                })
            except Exception:
                pass
            self._stop.wait(self.interval)


class CPUMonitor:
    """Background CPU / RAM sampler."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.samples: List[Dict] = []

    def start(self):
        self._stop.clear()
        self.samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> List[Dict]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.samples

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append({
                    "ts": time.perf_counter(),
                    "cpu_pct": psutil.cpu_percent(interval=None),
                    "ram_used_gb": psutil.virtual_memory().used / (1024 ** 3),
                    "ram_total_gb": psutil.virtual_memory().total / (1024 ** 3),
                })
            except Exception:
                pass
            self._stop.wait(self.interval)


# ============================================================
# BLEU Computation
# ============================================================

def compute_bleu_sacrebleu(refs: List[str], hyps: List[str],
                           target_lang: str = "en") -> Dict:
    """Compute BLEU scores using sacreBLEU (preferred, standard).

    Args:
        refs: reference translations
        hyps: hypothesis translations
        target_lang: target language code — 'en' uses tokenizer '13a',
                     'zh' uses tokenizer 'zh' (character-level).
    """
    # sacreBLEU tokenizer: '13a' for English/European, 'zh' for Chinese
    tokenizer = "zh" if target_lang == "zh" else "13a"
    bleu = SacreBLEU(tokenize=tokenizer)
    # sacreBLEU expects list of reference lists
    refs_nested = [[r] for r in refs]
    result = bleu.corpus_score(hyps, refs_nested)
    return {
        "bleu": round(result.score, 2),
        "bleu_1": round(result.counts[0] / max(result.totals[0], 1) * 100, 1),
        "bleu_2": round(result.counts[1] / max(result.totals[1], 1) * 100, 1),
        "bleu_3": round(result.counts[2] / max(result.totals[2], 1) * 100, 1),
        "bleu_4": round(result.counts[3] / max(result.totals[3], 1) * 100, 1),
        "sig": str(getattr(result, "signature", "")),
    }


def compute_bleu_nltk(refs: List[List[str]], hyps: List[str]) -> Dict:
    """Compute BLEU scores using NLTK (fallback)."""
    smooth = SmoothingFunction().method1
    weights = {
        "bleu_1": (1, 0, 0, 0),
        "bleu_2": (0.5, 0.5, 0, 0),
        "bleu_3": (0.33, 0.33, 0.33, 0),
        "bleu_4": (0.25, 0.25, 0.25, 0.25),
    }
    results = {}
    for name, w in weights.items():
        results[name] = round(
            corpus_bleu(refs, hyps, weights=w,
                        smoothing_function=smooth) * 100, 1
        )
    return results


def score_translations(references: List[str], hypotheses: List[str],
                       direction: str) -> Dict:
    """Score a set of translations and return a BLEU report.

    direction: 'zh→en' (target is English) or 'en→zh' (target is Chinese)
    """
    # Determine target language for proper tokenization
    tgt_lang = "en" if "en" in direction.split("→")[-1] else "zh"

    if _HAS_SACREBLEU:
        result = compute_bleu_sacrebleu(references, hypotheses, target_lang=tgt_lang)
    elif _HAS_NLTK:
        if tgt_lang == "zh":
            # Chinese: character-level tokenization
            refs_tokenized = [[list(r.replace(" ", ""))] for r in references]
            hyps_tokenized = [list(h.replace(" ", "")) for h in hypotheses]
        else:
            refs_tokenized = [[r.split()] for r in references]
            hyps_tokenized = [h.split() for h in hypotheses]
        result = compute_bleu_nltk(refs_tokenized, hyps_tokenized)
    else:
        result = {"error": "No BLEU library available"}
    result["direction"] = direction
    result["num_sentences"] = len(references)
    return result


# ============================================================
# Model Loading
# ============================================================

def load_opus_model(model_dir: str, device: str):
    """Load a MarianMT model + tokenizer."""
    from transformers import MarianMTModel, MarianTokenizer
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    tokenizer = MarianTokenizer.from_pretrained(str(path))
    model = MarianMTModel.from_pretrained(str(path)).to(device)
    model.eval()
    return model, tokenizer


# ============================================================
# 1. MODEL PERFORMANCE TEST — BLEU
# ============================================================

def test_bleu_scores(model, tokenizer, src_texts: List[str],
                     ref_texts: List[str], direction: str, device: str,
                     beam_size: int = 5) -> Dict:
    """Run BLEU evaluation on a set of parallel sentences."""
    print(f"\n{'='*60}")
    print(f" BLEU Evaluation: {direction} (beam={beam_size})")
    print(f" Sentences: {len(src_texts)}")
    print(f"{'='*60}")

    hypotheses = []
    latencies = []

    for i, text in enumerate(src_texts):
        t0 = time.perf_counter()
        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=cfg.max_length).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=beam_size,
                max_length=cfg.max_length,
                early_stopping=True,
            )
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        latencies.append((time.perf_counter() - t0) * 1000)
        hypotheses.append(translation.strip())

        if (i + 1) % 200 == 0:
            elapsed = sum(latencies[-200:]) / 200
            print(f"  [{i+1}/{len(src_texts)}] avg latency: {elapsed:.1f} ms")

    latencies_arr = np.array(latencies)
    bleu_result = score_translations(ref_texts, hypotheses, direction)

    print(f"\n  BLEU-4: {bleu_result.get('bleu_4', bleu_result.get('bleu', 'N/A'))}")
    print(f"  Latency — mean: {latencies_arr.mean():.1f} ms, "
          f"P50: {np.percentile(latencies_arr, 50):.1f} ms, "
          f"P95: {np.percentile(latencies_arr, 95):.1f} ms, "
          f"P99: {np.percentile(latencies_arr, 99):.1f} ms")

    bleu_result["latency_mean_ms"] = round(latencies_arr.mean(), 1)
    bleu_result["latency_p50_ms"] = round(np.percentile(latencies_arr, 50), 1)
    bleu_result["latency_p95_ms"] = round(np.percentile(latencies_arr, 95), 1)
    bleu_result["latency_p99_ms"] = round(np.percentile(latencies_arr, 99), 1)
    return bleu_result


# ============================================================
# 2. INFERENCE PERFORMANCE TEST
# ============================================================

@torch.no_grad()
def benchmark_single_latency(model, tokenizer, texts: List[str],
                              device: str, beam_size: int = 5,
                              warmup: int = 3, rounds: int = 10) -> Dict:
    """Measure single-sentence inference latency."""
    print(f"\n  --- Single-sentence latency (beam={beam_size}) ---")

    # Warmup
    for _ in range(warmup):
        inputs = tokenizer(texts[0], return_tensors="pt", padding=True,
                           truncation=True, max_length=cfg.max_length).to(device)
        _ = model.generate(**inputs, num_beams=beam_size,
                           max_length=cfg.max_length, early_stopping=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for text in texts[:rounds]:
        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=cfg.max_length).to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model.generate(**inputs, num_beams=beam_size,
                                 max_length=cfg.max_length, early_stopping=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    result = {
        "beam_size": beam_size,
        "mean_ms": round(arr.mean(), 1),
        "std_ms": round(arr.std(), 1),
        "min_ms": round(arr.min(), 1),
        "max_ms": round(arr.max(), 1),
        "p50_ms": round(np.percentile(arr, 50), 1),
        "p95_ms": round(np.percentile(arr, 95), 1),
    }
    for k, v in result.items():
        if k != "beam_size":
            print(f"    {k}: {v} ms")
    return result


@torch.no_grad()
def benchmark_batch_throughput(model, tokenizer, texts: List[str],
                                device: str, beam_size: int = 5,
                                batch_sizes: tuple = (1, 8, 32)) -> List[Dict]:
    """Measure batch inference throughput at various batch sizes."""
    results = []
    for bs in batch_sizes:
        if bs > len(texts):
            print(f"    batch={bs}: skipping (not enough texts)")
            continue

        batch_texts = texts[:bs]

        # Warmup
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=cfg.max_length).to(device)
        _ = model.generate(**inputs, num_beams=beam_size,
                           max_length=cfg.max_length, early_stopping=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark (average over multiple runs for small batches)
        n_runs = 20 if bs <= 8 else 5
        total_time = 0.0
        for _ in range(n_runs):
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=cfg.max_length).to(device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model.generate(**inputs, num_beams=beam_size,
                                     max_length=cfg.max_length, early_stopping=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_time += (time.perf_counter() - t0) * 1000

        avg_time = total_time / n_runs
        throughput = bs / (avg_time / 1000)  # samples / sec
        avg_per_sample = avg_time / bs

        result = {
            "batch_size": bs,
            "beam_size": beam_size,
            "total_ms": round(avg_time, 1),
            "ms_per_sample": round(avg_per_sample, 1),
            "throughput_samples_per_sec": round(throughput, 1),
        }
        results.append(result)
        print(f"    batch={bs:>3d} | total: {avg_time:>7.1f} ms | "
              f"per sample: {avg_per_sample:>6.1f} ms | "
              f"throughput: {throughput:>8.1f} samples/s")

    return results


def get_gpu_memory_snapshot() -> Dict:
    """Get current GPU memory usage."""
    if not _HAS_NVML:
        return {}
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "gpu_mem_used_mb": round(mem.used / (1024 ** 2), 1),
        "gpu_mem_total_mb": round(mem.total / (1024 ** 2), 1),
        "gpu_mem_free_mb": round(mem.free / (1024 ** 2), 1),
    }


# ============================================================
# Main Test Runner
# ============================================================

def run_all_tests(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'#'*60}")
    print(f"#  Machine Translation Model — Comprehensive Performance Test")
    print(f"#  Device: {device.upper()}")
    if torch.cuda.is_available():
        print(f"#  GPU: {torch.cuda.get_device_name(0)}")
        print(f"#  VRAM: {torch.cuda.get_device_properties(0).total_mem / (1024**3):.1f} GB")
    print(f"#  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    full_report = {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bleu_results": [],
        "inference_latency": [],
        "inference_throughput": [],
        "resource_usage": {},
    }

    # ── Load models ──
    print("\n\n[STEP 1] Loading models ...")
    gpu_mon = GPUMonitor()
    cpu_mon = CPUMonitor()

    gpu_mon.start()
    cpu_mon.start()

    print(f"  Loading zh→en model from: {cfg.opus_zh2en_dir}")
    zh2en_model, zh2en_tok = load_opus_model(cfg.opus_zh2en_dir, device)

    print(f"  Loading en→zh model from: {cfg.opus_en2zh_dir}")
    en2zh_model, en2zh_tok = load_opus_model(cfg.opus_en2zh_dir, device)

    # Record memory after loading both models
    gpu_samples = gpu_mon.stop()
    cpu_samples = cpu_mon.stop()

    if gpu_samples:
        peak_gpu_mem = max(s["mem_used_mb"] for s in gpu_samples)
        print(f"\n  GPU memory after loading 2 models: {peak_gpu_mem:.0f} MB used")
        full_report["resource_usage"]["model_load_gpu_mem_mb"] = round(peak_gpu_mem, 1)

    if cpu_samples:
        peak_ram = max(s["ram_used_gb"] for s in cpu_samples)
        print(f"  RAM after loading 2 models: {peak_ram:.1f} GB used")
        full_report["resource_usage"]["model_load_ram_gb"] = round(peak_ram, 1)

    # ── Load test data ──
    print(f"\n\n[STEP 2] Loading validation data ...")
    with open(cfg.valid_en_path, encoding="utf-8") as f:
        en_lines = [l.strip() for l in f if l.strip()]
    with open(cfg.valid_zh_path, encoding="utf-8") as f:
        zh_lines = [l.strip() for l in f if l.strip()]

    n_samples = min(cfg.bleu_sample_size, len(en_lines), len(zh_lines))
    en_subset = en_lines[:n_samples]
    zh_subset = zh_lines[:n_samples]
    print(f"  Loaded {len(en_lines)} EN + {len(zh_lines)} ZH lines, "
          f"using {n_samples} for BLEU test")

    # ── Test 1: BLEU Scores ──
    if not args.skip_bleu:
        print(f"\n\n[STEP 3] Model Performance — BLEU Scores")
        print("-" * 60)

        # zh→en
        result_z2e = test_bleu_scores(
            zh2en_model, zh2en_tok,
            zh_subset, en_subset,  # source=zh, reference=en
            "zh→en", device, beam_size=5
        )
        full_report["bleu_results"].append(result_z2e)

        # en→zh
        result_e2z = test_bleu_scores(
            en2zh_model, en2zh_tok,
            en_subset, zh_subset,  # source=en, reference=zh
            "en→zh", device, beam_size=5
        )
        full_report["bleu_results"].append(result_e2z)

    # ── Test 2: Inference Performance ──
    print(f"\n\n[STEP 4] Inference Performance Benchmark")
    print("-" * 60)

    # Use a mix of short and long sentences for representative benchmarks
    bench_texts_en = en_lines[:100]
    bench_texts_zh = zh_lines[:100]

    # 4a. Single-sentence latency (various beam sizes)
    print("\n  [4a] Single-sentence latency vs beam size (zh→en model):")
    for bs in cfg.beam_sizes:
        result = benchmark_single_latency(
            zh2en_model, zh2en_tok, bench_texts_zh,
            device, beam_size=bs,
            warmup=cfg.warmup_rounds, rounds=cfg.benchmark_rounds
        )
        full_report["inference_latency"].append(result)

    print("\n  [4a'] Single-sentence latency vs beam size (en→zh model):")
    for bs in cfg.beam_sizes:
        result = benchmark_single_latency(
            en2zh_model, en2zh_tok, bench_texts_en,
            device, beam_size=bs,
            warmup=cfg.warmup_rounds, rounds=cfg.benchmark_rounds
        )
        full_report["inference_latency"].append(result)

    # 4b. Batch throughput
    print("\n  [4b] Batch throughput at different batch sizes (beam=6):")
    for label, model, tok, src_texts in [
        ("zh→en", zh2en_model, zh2en_tok, bench_texts_zh),
        ("en→zh", en2zh_model, en2zh_tok, bench_texts_en),
    ]:
        print(f"\n  --- {label} ---")
        batch_results = benchmark_batch_throughput(
            model, tok, src_texts,
            device, beam_size=6, batch_sizes=cfg.batch_sizes
        )
        for r in batch_results:
            r["direction"] = label
        full_report["inference_throughput"].extend(batch_results)

    # 4c. GPU/CPU resource monitoring during sustained inference
    print(f"\n\n[STEP 5] Resource Usage Under Sustained Load")
    print("-" * 60)

    gpu_mon2 = GPUMonitor(interval=0.05)
    cpu_mon2 = CPUMonitor(interval=0.2)

    # Run continuous inference for ~30 seconds
    gpu_mon2.start()
    cpu_mon2.start()

    print("  Running sustained inference for 30 seconds ...")
    t_start = time.perf_counter()
    n_inferences = 0
    idx = 0
    while time.perf_counter() - t_start < 30:
        text = bench_texts_zh[idx % len(bench_texts_zh)]
        inputs = zh2en_tok(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=cfg.max_length).to(device)
        with torch.no_grad():
            _ = zh2en_model.generate(
                **inputs, num_beams=5, max_length=cfg.max_length,
                early_stopping=True
            )
        n_inferences += 1
        idx += 1

    gpu_sustained = gpu_mon2.stop()
    cpu_sustained = cpu_mon2.stop()

    if gpu_sustained:
        gpu_utils = [s["gpu_util_pct"] for s in gpu_sustained]
        mem_useds = [s["mem_used_mb"] for s in gpu_sustained]
        peak_mem = max(mem_useds)
        # Exclude first few samples (may contain idle)
        active_utils = gpu_utils[10:] if len(gpu_utils) > 10 else gpu_utils

        print(f"  GPU Utilization — mean: {np.mean(active_utils):.1f}%, "
              f"peak: {np.max(active_utils):.1f}%")
        print(f"  GPU Memory — peak: {peak_mem:.0f} MB")

        full_report["resource_usage"]["sustained_gpu"] = {
            "gpu_util_mean_pct": round(np.mean(active_utils), 1),
            "gpu_util_peak_pct": round(np.max(active_utils), 1),
            "gpu_mem_peak_mb": round(peak_mem, 1),
            "inferences_completed": n_inferences,
            "duration_s": 30,
        }

    if cpu_sustained:
        cpu_vals = [s["cpu_pct"] for s in cpu_sustained[5:]]
        ram_vals = [s["ram_used_gb"] for s in cpu_sustained]
        print(f"  CPU Utilization — mean: {np.mean(cpu_vals):.1f}%, "
              f"peak: {np.max(cpu_vals):.1f}%")
        print(f"  RAM — peak: {np.max(ram_vals):.1f} GB")

        full_report["resource_usage"]["sustained_cpu"] = {
            "cpu_util_mean_pct": round(np.mean(cpu_vals), 1),
            "cpu_util_peak_pct": round(np.max(cpu_vals), 1),
            "ram_peak_gb": round(np.max(ram_vals), 1),
        }

    # ── Summary ──
    print(f"\n\n{'#'*60}")
    print(f"#  TEST SUMMARY")
    print(f"{'#'*60}")

    if not args.skip_bleu:
        for r in full_report["bleu_results"]:
            print(f"\n  {r['direction']}:")
            for k in ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "bleu"]:
                if k in r:
                    print(f"    {k}: {r[k]}")
            print(f"    latency_mean: {r.get('latency_mean_ms', 'N/A')} ms")
            print(f"    latency_p95: {r.get('latency_p95_ms', 'N/A')} ms")

    print(f"\n  Inference Latency (beam=5, zh→en):")
    for r in full_report["inference_latency"]:
        if r.get("beam_size") == 5:
            print(f"    mean: {r['mean_ms']} ms | p95: {r['p95_ms']} ms | "
                  f"std: {r['std_ms']} ms")

    print(f"\n  Batch Throughput (beam=6):")
    for r in full_report["inference_throughput"]:
        print(f"    {r.get('direction', ''):>5s} batch={r['batch_size']:>3d} | "
              f"{r['throughput_samples_per_sec']:>8.1f} samples/s | "
              f"{r['ms_per_sample']:>6.1f} ms/sample")

    res = full_report.get("resource_usage", {})
    if "sustained_gpu" in res:
        g = res["sustained_gpu"]
        print(f"\n  GPU under sustained load:")
        print(f"    utilization: {g['gpu_util_mean_pct']}% mean / "
              f"{g['gpu_util_peak_pct']}% peak")
        print(f"    memory peak: {g['gpu_mem_peak_mb']} MB")
    if "sustained_cpu" in res:
        c = res["sustained_cpu"]
        print(f"  CPU under sustained load:")
        print(f"    utilization: {c['cpu_util_mean_pct']}% mean / "
              f"{c['cpu_util_peak_pct']}% peak")

    # ── Save report ──
    if args.output:
        output_path = args.output
    else:
        output_path = f"benchmark_report_{time.strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    print(f"\n  Full report saved to: {output_path}")

    # Also update the project report with actual values
    print(f"\n{'='*60}")
    print(f"  You can compare these results with values in 项目报告.md")
    print(f"  and update the report using write_report_p4.py if needed.")
    print(f"{'='*60}")

    return full_report


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive MT model performance & inference benchmark"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 200 sentences for BLEU")
    parser.add_argument("--skip-bleu", action="store_true",
                        help="Skip BLEU evaluation (inference benchmark only)")
    parser.add_argument("--output", type=str, default="",
                        help="Output JSON report path")
    parser.add_argument("--bleu-samples", type=int, default=2000,
                        help="Number of sentences for BLEU test (default: 2000)")
    parser.add_argument("--beam", type=int, default=0,
                        help="Override beam size for BLEU (default: 5)")
    args = parser.parse_args()

    if args.quick:
        cfg.bleu_sample_size = 200
        cfg.benchmark_rounds = 5
    else:
        cfg.bleu_sample_size = args.bleu_samples

    report = run_all_tests(args)

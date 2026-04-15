"""
Benchmark v3: DAG-based datasets.

Measures forward passes to generate valid+diverse sequences.
Compares against theoretical minimum (critical path + 1).
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from collections import Counter
from datasets_v3 import ALL_V3_DATASETS
from common import TinyTransformer
from benchmark import (
    train_causal, train_bidirectional,
    method_autoregressive, method_mask_predict_adaptive, method_speculative,
)
from entropy_aware import method_sampling_mask_predict, method_bidir_speculative


NUM_SAMPLES = 200


def evaluate(results, steps, dataset, method_name):
    num_samples = len(results)
    valid = sum(1 for r in results if list(r) in dataset["sequences"])
    counter = Counter(results)
    probs = [c / num_samples for c in counter.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(max(len(dataset["sequences"]), 1))
    avg_steps = sum(steps) / len(steps)

    return {
        "method": method_name,
        "valid_pct": 100 * valid / num_samples,
        "diversity_entropy": entropy,
        "max_entropy": max_entropy,
        "entropy_ratio": entropy / max_entropy if max_entropy > 0 else 0,
        "avg_steps": avg_steps,
        "min_steps_theory": dataset["min_steps"],
        "step_ratio": avg_steps / dataset["min_steps"] if dataset["min_steps"] > 0 else avg_steps,
    }


if __name__ == "__main__":
    torch.manual_seed(42)

    methods_order = ["AR", "MP-Ad", "Spec", "Samp-MP", "Bidir-Spec"]
    all_results = []

    for label, ds_fn in ALL_V3_DATASETS:
        ds = ds_fn()
        print(f"\n{'='*70}")
        print(f"  {label}  |  seqs={len(ds['sequences'])}  depth={ds['sequential_depth']}  "
              f"min_steps={ds['min_steps']}")
        schedule = ds["schedule"]
        sched_str = " → ".join(
            "{" + ",".join(str(p) for p in sorted(s)) + "}" for s in schedule
        )
        print(f"  Schedule: {sched_str}")
        print(f"{'='*70}")

        # Train
        print("  Training causal...", end=" ", flush=True)
        causal = train_causal(ds, num_steps=3000)
        print("bidir...", end=" ", flush=True)
        bidir = train_bidirectional(ds, num_steps=5000)
        print("done")

        ds_results = {}

        # AR
        r, s = method_autoregressive(causal, ds, NUM_SAMPLES)
        ds_results["AR"] = evaluate(r, s, ds, "AR")

        # MP-Adaptive
        r, s = method_mask_predict_adaptive(bidir, ds, NUM_SAMPLES)
        ds_results["MP-Ad"] = evaluate(r, s, ds, "MP-Ad")

        # Speculative
        r, s = method_speculative(causal, ds, NUM_SAMPLES)
        ds_results["Spec"] = evaluate(r, s, ds, "Spec")

        # Sampling MP
        r, s = method_sampling_mask_predict(bidir, ds, NUM_SAMPLES)
        ds_results["Samp-MP"] = evaluate(r, s, ds, "Samp-MP")

        # Bidir Speculative
        r, s = method_bidir_speculative(causal, bidir, ds, NUM_SAMPLES)
        ds_results["Bidir-Spec"] = evaluate(r, s, ds, "Bidir-Spec")

        # Print
        print(f"\n  {'Method':12s} {'Steps':>7s} {'Ratio':>7s} {'Valid%':>7s} "
              f"{'Entropy':>8s} {'E-ratio':>8s}")
        print(f"  {'-'*53}")
        for m in methods_order:
            ev = ds_results[m]
            print(f"  {m:12s} {ev['avg_steps']:6.2f} {ev['step_ratio']:6.2f}x "
                  f"{ev['valid_pct']:5.1f}% {ev['diversity_entropy']:7.3f} "
                  f"{ev['entropy_ratio']:7.1%}")

        all_results.append((label, ds, ds_results))

    # ── Grand summary ──
    print(f"\n\n{'='*100}")
    print("GRAND SUMMARY")
    print(f"{'='*100}")
    print(f"\n{'Dataset':22s} {'min':>4s}", end="")
    for m in methods_order:
        print(f"  {m:>11s}", end="")
    print()
    print(f"{'':22s} {'step':>4s}", end="")
    for m in methods_order:
        print(f"  {'stp  v%  e%':>11s}", end="")
    print()
    print("-" * (27 + 13 * len(methods_order)))

    for label, ds, results in all_results:
        min_s = ds["min_steps"]
        print(f"{label:22s} {min_s:>4d}", end="")
        for m in methods_order:
            ev = results[m]
            v_mark = " " if ev["valid_pct"] >= 99 else "!"
            print(f"  {ev['avg_steps']:4.1f} {ev['valid_pct']:3.0f}{v_mark} "
                  f"{ev['entropy_ratio']:3.0%}", end="")
        print()

    print(f"\nFormat: steps  valid%  entropy_ratio")
    print(f"! = validity < 99%")

    # Efficiency summary
    print(f"\n\n{'='*80}")
    print("EFFICIENCY: avg_steps / min_steps (lower=better, 1.0=optimal)")
    print(f"{'='*80}")
    print(f"{'Dataset':22s} {'min':>4s}", end="")
    for m in methods_order:
        print(f" {m:>10s}", end="")
    print()
    print("-" * (27 + 11 * len(methods_order)))

    for label, ds, results in all_results:
        min_s = ds["min_steps"]
        print(f"{label:22s} {min_s:>4d}", end="")
        for m in methods_order:
            ev = results[m]
            ratio = ev["step_ratio"]
            marker = "*" if ev["valid_pct"] < 99 or ev["entropy_ratio"] < 0.5 else " "
            print(f" {ratio:>8.2f}x{marker}", end="")
        print()

    print(f"\n* = invalid (<99% valid) or low diversity (<50% entropy ratio)")

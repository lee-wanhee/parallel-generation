"""
Benchmark v2: Random diverse datasets with shared vocabulary.

Runs all methods across multiple random seeds to get robust results.
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from collections import Counter
from datasets_v2 import (
    dataset_v2_independent,
    dataset_v2_depth1_narrow,
    dataset_v2_depth1_wide,
    dataset_v2_depth2,
    dataset_v2_depth2_wide,
    dataset_v2_depth3,
    dataset_v2_depth1_highbranch,
    dataset_v2_depth2_highbranch,
)
from common import TinyTransformer
from benchmark import (
    train_causal, train_bidirectional,
    method_autoregressive, method_jacobi,
    method_mask_predict_adaptive, method_speculative,
    evaluate,
)
from entropy_aware import method_sampling_mask_predict, method_bidir_speculative
from diffusion import train_diffusion, diffusion_inference_adaptive


NUM_SAMPLES = 200
NUM_SEEDS = 3
SEEDS = [42, 123, 999]


def run_one_dataset(ds_fn, seed):
    """Run all methods on one dataset with one seed."""
    ds = ds_fn(seed=seed)
    torch.manual_seed(seed)

    # Train models
    causal = train_causal(ds, num_steps=3000)
    bidir = train_bidirectional(ds, num_steps=5000)

    results = {}

    # AR
    r, s = method_autoregressive(causal, ds, NUM_SAMPLES)
    results["AR"] = evaluate(r, s, ds, "AR")

    # Jacobi argmax
    r, s = method_jacobi(causal, ds, NUM_SAMPLES, use_sampling=False)
    results["Jacobi-A"] = evaluate(r, s, ds, "Jacobi-A")

    # MP-Adaptive
    r, s = method_mask_predict_adaptive(bidir, ds, NUM_SAMPLES)
    results["MP-Ad"] = evaluate(r, s, ds, "MP-Ad")

    # Speculative
    r, s = method_speculative(causal, ds, NUM_SAMPLES)
    results["Spec"] = evaluate(r, s, ds, "Spec")

    # Sampling MP (new)
    r, s = method_sampling_mask_predict(bidir, ds, NUM_SAMPLES)
    results["Samp-MP"] = evaluate(r, s, ds, "Samp-MP")

    # Bidir Speculative (new)
    r, s = method_bidir_speculative(causal, bidir, ds, NUM_SAMPLES)
    results["Bidir-Spec"] = evaluate(r, s, ds, "Bidir-Spec")

    # Diffusion
    diff_model = train_diffusion(ds, num_steps=3000)
    r, s = diffusion_inference_adaptive(diff_model, ds, NUM_SAMPLES)
    results["Diff"] = evaluate(r, s, ds, "Diff")

    return ds, results


def aggregate_results(all_runs):
    """Average results across seeds."""
    methods = list(all_runs[0][1].keys())
    agg = {}
    for method in methods:
        valid_pcts = [run[1][method]["valid_pct"] for run in all_runs]
        steps = [run[1][method]["avg_steps"] for run in all_runs]
        entropies = [run[1][method]["diversity_entropy"] for run in all_runs]
        max_ent = all_runs[0][1][method]["max_entropy"]

        agg[method] = {
            "valid_mean": np.mean(valid_pcts),
            "valid_std": np.std(valid_pcts),
            "steps_mean": np.mean(steps),
            "steps_std": np.std(steps),
            "entropy_mean": np.mean(entropies),
            "entropy_std": np.std(entropies),
            "max_entropy": max_ent,
        }
    return agg


if __name__ == "__main__":
    dataset_configs = [
        ("independent (d=0)", dataset_v2_independent),
        ("d1_narrow (d=1)", dataset_v2_depth1_narrow),
        ("d1_wide (d=1)", dataset_v2_depth1_wide),
        ("d2 (d=2)", dataset_v2_depth2),
        ("d2_wide (d=2)", dataset_v2_depth2_wide),
        ("d3 (d=3)", dataset_v2_depth3),
        ("d1_hb (d=1,b=4)", dataset_v2_depth1_highbranch),
        ("d2_hb (d=2,b=3)", dataset_v2_depth2_highbranch),
    ]

    all_aggregated = []
    methods_order = ["AR", "Jacobi-A", "MP-Ad", "Spec", "Samp-MP", "Bidir-Spec", "Diff"]

    for ds_label, ds_fn in dataset_configs:
        print(f"\n{'='*70}")
        print(f"  {ds_label}")
        print(f"{'='*70}")

        runs = []
        for seed in SEEDS:
            print(f"  seed={seed}...", end=" ", flush=True)
            ds, results = run_one_dataset(ds_fn, seed)
            runs.append((ds, results))
            print(f"done")

        agg = aggregate_results(runs)
        all_aggregated.append((ds_label, runs[0][0]["sequential_depth"], agg))

        # Print per-dataset summary
        print(f"\n  {'Method':12s} {'Valid%':>8s} {'Steps':>10s} {'Entropy':>12s}")
        print(f"  {'-'*46}")
        for m in methods_order:
            a = agg[m]
            print(f"  {m:12s} {a['valid_mean']:5.1f}±{a['valid_std']:3.1f} "
                  f"{a['steps_mean']:5.2f}±{a['steps_std']:4.2f} "
                  f"{a['entropy_mean']:5.3f}±{a['entropy_std']:4.3f} /{a['max_entropy']:.2f}")

    # ── Grand summary ──
    print(f"\n\n{'='*100}")
    print("GRAND SUMMARY (averaged over 3 seeds)")
    print(f"{'='*100}")

    # Steps table
    print(f"\n--- Average Steps (lower=better) ---")
    print(f"{'Dataset':20s} {'d':>3s}", end="")
    for m in methods_order:
        print(f" {m:>10s}", end="")
    print()
    print("-" * (24 + 11 * len(methods_order)))
    for label, depth, agg in all_aggregated:
        print(f"{label:20s} {depth:>3s}", end="")
        for m in methods_order:
            v = agg[m]
            marker = "*" if v["valid_mean"] < 99 else " "
            print(f" {v['steps_mean']:>8.2f}{marker}", end="")
        print()
    print("* = validity < 99%")

    # Validity table
    print(f"\n--- Validity % ---")
    print(f"{'Dataset':20s} {'d':>3s}", end="")
    for m in methods_order:
        print(f" {m:>10s}", end="")
    print()
    print("-" * (24 + 11 * len(methods_order)))
    for label, depth, agg in all_aggregated:
        print(f"{label:20s} {depth:>3s}", end="")
        for m in methods_order:
            v = agg[m]
            print(f" {v['valid_mean']:>8.1f}%", end="")
        print()

    # Diversity table
    print(f"\n--- Diversity (entropy bits, higher=better) ---")
    print(f"{'Dataset':20s} {'max':>5s}", end="")
    for m in methods_order:
        print(f" {m:>10s}", end="")
    print()
    print("-" * (26 + 11 * len(methods_order)))
    for label, depth, agg in all_aggregated:
        max_e = f"{agg['AR']['max_entropy']:.2f}"
        print(f"{label:20s} {max_e:>5s}", end="")
        for m in methods_order:
            v = agg[m]
            print(f" {v['entropy_mean']:>9.3f}", end="")
        print()

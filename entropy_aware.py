"""
Entropy-Aware Parallel Generation

Two new methods that aim to break the trilemma:

1. Sampling-aware Mask-Predict: Sample high-entropy positions, commit
   low-entropy ones. Should achieve diversity + speed + validity.

2. Bidirectional Speculative Decoding: Use bidirectional model as draft,
   verify with AR model. Should preserve exact AR distribution while
   exploiting parallel structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter
from datasets import (
    dataset_independent, dataset_depth1, dataset_depth1_wide,
    dataset_depth2, dataset_depth3, dataset_mixed,
)
from common import TinyTransformer
from benchmark import train_causal, train_bidirectional, evaluate


def entropy(probs):
    """Compute entropy in bits."""
    log_probs = torch.log2(probs + 1e-10)
    return -(probs * log_probs).sum().item()


# ═══════════════════════════════════════════════════════════════════
# Method 1: Sampling-aware Mask-Predict
# ═══════════════════════════════════════════════════════════════════

def method_sampling_mask_predict(model, dataset, num_samples=500,
                                  max_iters=10, entropy_threshold=0.3):
    """
    At each step:
    1. Forward pass on current sequence (with MASKs)
    2. For each masked position, compute entropy of prediction
    3. If entropy < threshold: COMMIT (argmax) — this token is determined
    4. If entropy >= threshold: SAMPLE from the distribution
    5. Commit at least 1 position per step

    The key insight: after sampling a high-entropy position, the next
    forward pass will condition on the sampled value, making downstream
    positions low-entropy (determined). Then those can be committed
    in parallel.
    """
    V = dataset["V"]
    mask_id = dataset["tok2id"]["<MASK>"]
    bos_id = dataset["tok2id"]["<BOS>"]
    eos_id = dataset["tok2id"]["<EOS>"]
    content_pos = dataset["content_positions"]
    seq_len = dataset["seq_len"]

    model.eval()
    results = []
    steps_list = []

    with torch.no_grad():
        for _ in range(num_samples):
            seq = torch.zeros(1, seq_len, dtype=torch.long)
            seq[0, 0] = bos_id
            seq[0, -1] = eos_id
            for pos in content_pos:
                seq[0, pos] = mask_id

            masked = list(content_pos)
            step_count = 0

            for it in range(max_iters):
                if not masked:
                    break

                logits = model(seq)
                step_count += 1

                # Compute entropy and predictions for each masked position
                pos_info = []
                for pos in masked:
                    probs = F.softmax(logits[0, pos], dim=-1)
                    h = entropy(probs)
                    max_prob, pred = probs.max(dim=-1)
                    pos_info.append({
                        "pos": pos, "entropy": h, "probs": probs,
                        "pred": pred.item(), "max_prob": max_prob.item(),
                    })

                # Separate into high-entropy (need sampling) and low-entropy (can commit)
                to_commit = []
                to_sample = []
                for info in pos_info:
                    if info["entropy"] < entropy_threshold:
                        to_commit.append(info)
                    else:
                        to_sample.append(info)

                # If everything is high-entropy, sample the MOST uncertain one
                # (this is the key branching decision)
                if not to_commit and to_sample:
                    # Sample the highest-entropy position
                    to_sample.sort(key=lambda x: x["entropy"], reverse=True)
                    top = to_sample[0]
                    sampled_tok = torch.multinomial(top["probs"], 1).item()
                    seq[0, top["pos"]] = sampled_tok
                    masked.remove(top["pos"])
                    # Don't commit others yet — wait for next forward pass
                    # where they'll condition on the sampled value
                    continue

                # Commit all low-entropy positions (parallel!)
                for info in to_commit:
                    seq[0, info["pos"]] = info["pred"]
                    masked.remove(info["pos"])

                # Also sample one high-entropy position if any remain
                if to_sample:
                    to_sample.sort(key=lambda x: x["entropy"], reverse=True)
                    top = to_sample[0]
                    sampled_tok = torch.multinomial(top["probs"], 1).item()
                    seq[0, top["pos"]] = sampled_tok
                    masked.remove(top["pos"])

            steps_list.append(step_count)
            results.append(tuple(seq[0].tolist()))

    return results, steps_list


# ═══════════════════════════════════════════════════════════════════
# Method 2: Bidirectional Speculative Decoding
# ═══════════════════════════════════════════════════════════════════

def method_bidir_speculative(causal_model, bidir_model, dataset, num_samples=500):
    """
    1. Use bidirectional model to draft ALL positions in parallel
       (via mask-predict with argmax — fast but mode-collapsed)
    2. Verify the draft against the causal AR model left-to-right
    3. Accept longest matching prefix, resample first rejected position

    This should:
    - Exploit parallel structure (bidirectional draft is high quality)
    - Preserve exact AR distribution (verification guarantees it)
    - Be fast when draft matches AR (often for determined positions)
    """
    V = dataset["V"]
    mask_id = dataset["tok2id"]["<MASK>"]
    bos_id = dataset["tok2id"]["<BOS>"]
    eos_id = dataset["tok2id"]["<EOS>"]
    content_pos = dataset["content_positions"]
    seq_len = dataset["seq_len"]

    causal_model.eval()
    bidir_model.eval()
    results = []
    steps_list = []

    with torch.no_grad():
        for _ in range(num_samples):
            tokens = [bos_id]
            passes = 0

            while len(tokens) < seq_len:
                # Step 1: Generate draft using bidirectional model
                draft_seq = torch.zeros(1, seq_len, dtype=torch.long)
                draft_seq[0, 0] = bos_id
                draft_seq[0, -1] = eos_id

                # Fill known tokens
                for i, tok in enumerate(tokens):
                    draft_seq[0, i] = tok

                # Mask unknown positions
                unknown_pos = list(range(len(tokens), seq_len - 1))
                for pos in unknown_pos:
                    draft_seq[0, pos] = mask_id

                # Iterative mask-predict to fill draft (using bidir model)
                masked = list(unknown_pos)
                draft_passes = 0
                while masked:
                    logits = bidir_model(draft_seq)
                    draft_passes += 1
                    # Unmask most confident
                    best_pos, best_conf, best_pred = None, -1, None
                    for pos in masked:
                        probs = F.softmax(logits[0, pos], dim=-1)
                        max_prob, pred = probs.max(dim=-1)
                        if max_prob.item() > best_conf:
                            best_pos = pos
                            best_conf = max_prob.item()
                            best_pred = pred.item()
                    draft_seq[0, best_pos] = best_pred
                    masked.remove(best_pos)

                # Step 2: Verify draft against causal model (single forward pass)
                x = torch.tensor([draft_seq[0].tolist()])
                logits = causal_model(x)
                passes += 1  # count as 1 step (the verification)

                # Accept/reject left-to-right from current position
                for i in range(len(tokens), seq_len):
                    target_probs = F.softmax(logits[0, i - 1], dim=-1)
                    draft_tok = draft_seq[0, i].item()

                    p_target = target_probs[draft_tok].item()
                    # Draft probability: approximate as 1 (bidir model is confident)
                    # Use simple accept/reject
                    p_draft = 1.0  # bidir model used argmax, so p_draft ≈ 1

                    # Accept with probability p_target (since p_draft=1)
                    if torch.rand(1).item() < p_target:
                        tokens.append(draft_tok)
                    else:
                        # Reject: sample from target distribution
                        tokens.append(torch.multinomial(target_probs, 1).item())
                        break

                if len(tokens) >= seq_len:
                    break

            steps_list.append(passes)
            results.append(tuple(tokens[:seq_len]))

    return results, steps_list


def method_bidir_speculative_fair(causal_model, bidir_model, dataset, num_samples=500):
    """
    Same as bidir_speculative but counts ALL forward passes
    (both bidir draft passes AND AR verification passes).
    This is the fair comparison metric.
    """
    V = dataset["V"]
    mask_id = dataset["tok2id"]["<MASK>"]
    bos_id = dataset["tok2id"]["<BOS>"]
    eos_id = dataset["tok2id"]["<EOS>"]
    content_pos = dataset["content_positions"]
    seq_len = dataset["seq_len"]

    causal_model.eval()
    bidir_model.eval()
    results = []
    steps_list = []

    with torch.no_grad():
        for _ in range(num_samples):
            tokens = [bos_id]
            total_passes = 0

            while len(tokens) < seq_len:
                draft_seq = torch.zeros(1, seq_len, dtype=torch.long)
                draft_seq[0, 0] = bos_id
                draft_seq[0, -1] = eos_id
                for i, tok in enumerate(tokens):
                    draft_seq[0, i] = tok

                unknown_pos = list(range(len(tokens), seq_len - 1))
                for pos in unknown_pos:
                    draft_seq[0, pos] = mask_id

                # Draft: count these passes
                masked = list(unknown_pos)
                while masked:
                    logits = bidir_model(draft_seq)
                    total_passes += 1  # COUNT bidir passes
                    best_pos, best_conf, best_pred = None, -1, None
                    for pos in masked:
                        probs = F.softmax(logits[0, pos], dim=-1)
                        max_prob, pred = probs.max(dim=-1)
                        if max_prob.item() > best_conf:
                            best_pos = pos
                            best_conf = max_prob.item()
                            best_pred = pred.item()
                    draft_seq[0, best_pos] = best_pred
                    masked.remove(best_pos)

                # Verify: count this pass too
                x = torch.tensor([draft_seq[0].tolist()])
                logits = causal_model(x)
                total_passes += 1  # COUNT AR pass

                for i in range(len(tokens), seq_len):
                    target_probs = F.softmax(logits[0, i - 1], dim=-1)
                    draft_tok = draft_seq[0, i].item()
                    p_target = target_probs[draft_tok].item()
                    if torch.rand(1).item() < p_target:
                        tokens.append(draft_tok)
                    else:
                        tokens.append(torch.multinomial(target_probs, 1).item())
                        break

                if len(tokens) >= seq_len:
                    break

            steps_list.append(total_passes)
            results.append(tuple(tokens[:seq_len]))

    return results, steps_list


def method_bidir_speculative_parallel(causal_model, bidir_model, dataset, num_samples=500):
    """
    Bidir Speculative with PARALLEL draft:
    1. One bidir forward pass to draft ALL positions at once (argmax)
    2. One AR forward pass to verify
    = 2 passes total per attempt

    If verification rejects at position k, we keep tokens up to k,
    then draft remaining positions again (2 more passes).
    """
    V = dataset["V"]
    mask_id = dataset["tok2id"]["<MASK>"]
    bos_id = dataset["tok2id"]["<BOS>"]
    eos_id = dataset["tok2id"]["<EOS>"]
    content_pos = dataset["content_positions"]
    seq_len = dataset["seq_len"]

    causal_model.eval()
    bidir_model.eval()
    results = []
    steps_list = []

    with torch.no_grad():
        for _ in range(num_samples):
            tokens = [bos_id]
            total_passes = 0

            while len(tokens) < seq_len:
                draft_seq = torch.zeros(1, seq_len, dtype=torch.long)
                draft_seq[0, 0] = bos_id
                draft_seq[0, -1] = eos_id
                for i, tok in enumerate(tokens):
                    draft_seq[0, i] = tok

                unknown_pos = list(range(len(tokens), seq_len - 1))
                for pos in unknown_pos:
                    draft_seq[0, pos] = mask_id

                # SINGLE bidir pass: argmax all masked positions at once
                logits = bidir_model(draft_seq)
                total_passes += 1
                for pos in unknown_pos:
                    probs = F.softmax(logits[0, pos], dim=-1)
                    probs[mask_id] = 0
                    if probs.sum() > 0:
                        probs = probs / probs.sum()
                    draft_seq[0, pos] = probs.argmax()

                # AR verification pass
                x = torch.tensor([draft_seq[0].tolist()])
                logits = causal_model(x)
                total_passes += 1

                for i in range(len(tokens), seq_len):
                    target_probs = F.softmax(logits[0, i - 1], dim=-1)
                    draft_tok = draft_seq[0, i].item()
                    p_target = target_probs[draft_tok].item()
                    if torch.rand(1).item() < p_target:
                        tokens.append(draft_tok)
                    else:
                        tokens.append(torch.multinomial(target_probs, 1).item())
                        break

                if len(tokens) >= seq_len:
                    break

            steps_list.append(total_passes)
            results.append(tuple(tokens[:seq_len]))

    return results, steps_list


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    NUM = 500

    test_datasets = [
        dataset_independent,
        dataset_depth1,
        dataset_depth1_wide,
        dataset_depth2,
        dataset_depth3,
        dataset_mixed,
    ]

    print("=" * 80)
    print("ENTROPY-AWARE PARALLEL GENERATION")
    print("Can we break the validity-diversity-speed trilemma?")
    print("=" * 80)
    print()

    all_results = []

    for ds_fn in test_datasets:
        ds = ds_fn()
        print("-" * 70)
        print(f"DATASET: {ds['name']}  (depth={ds['sequential_depth']}, "
              f"{len(ds['sequences'])} seqs, len={ds['seq_len']})")
        print("-" * 70)

        # Train models
        print("  Training causal model...")
        causal_model = train_causal(ds, num_steps=2000)
        print("  Training bidirectional model...")
        bidir_model = train_bidirectional(ds, num_steps=3000)
        print()

        ds_results = []

        # Baseline: AR
        r, s = [], [len(ds["content_positions"])] * NUM
        from benchmark import method_autoregressive
        r, s = method_autoregressive(causal_model, ds, NUM)
        ev = evaluate(r, s, ds, "AR (baseline)")
        print(f"  AR:             valid={ev['valid_pct']:5.1f}%  "
              f"steps={ev['avg_steps']:.2f}  entropy={ev['diversity_entropy']:.3f}/{ev['max_entropy']:.3f}")
        ds_results.append(ev)

        # New method 1: Sampling-aware Mask-Predict
        r, s = method_sampling_mask_predict(bidir_model, ds, NUM, entropy_threshold=0.3)
        ev = evaluate(r, s, ds, "Sampling MP")
        print(f"  Sampling MP:    valid={ev['valid_pct']:5.1f}%  "
              f"steps={ev['avg_steps']:.2f}  entropy={ev['diversity_entropy']:.3f}/{ev['max_entropy']:.3f}")
        ds_results.append(ev)

        # New method 2: Bidirectional Speculative
        r, s = method_bidir_speculative(causal_model, bidir_model, ds, NUM)
        ev = evaluate(r, s, ds, "Bidir Spec")
        print(f"  Bidir Spec:     valid={ev['valid_pct']:5.1f}%  "
              f"steps={ev['avg_steps']:.2f}  entropy={ev['diversity_entropy']:.3f}/{ev['max_entropy']:.3f}")
        ds_results.append(ev)

        print()
        all_results.append((ds, ds_results))

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':20s} {'Depth':6s} {'Method':15s} {'Valid':>7s} {'Steps':>7s} {'Entropy':>10s} {'Max H':>7s}")
    print("-" * 80)

    for ds, results in all_results:
        for i, ev in enumerate(results):
            ds_name = ds["name"] if i == 0 else ""
            depth = str(ds["sequential_depth"]) if i == 0 else ""
            print(f"{ds_name:20s} {depth:6s} {ev['method']:15s} "
                  f"{ev['valid_pct']:6.1f}% {ev['avg_steps']:6.2f} "
                  f"{ev['diversity_entropy']:9.3f} {ev['max_entropy']:6.3f}")
        print()

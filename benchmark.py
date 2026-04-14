"""
Comprehensive benchmark: Steps to resolve uncertainty

For each dataset × method, measures:
1. Number of forward passes (steps) to produce a valid sequence
2. Validity rate
3. Diversity (entropy of output distribution)
4. Comparison against theoretical minimum steps = sequential depth + 1

The key question: which method resolves uncertainty in the fewest steps?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter
from datasets import (
    dataset_independent, dataset_depth1, dataset_depth1_wide,
    dataset_depth2, dataset_depth2_wide, dataset_depth3, dataset_mixed,
    ALL_DATASETS,
)
from common import TinyTransformer


# ═══════════════════════════════════════════════════════════════════
# Model training helpers
# ═══════════════════════════════════════════════════════════════════

def train_causal(dataset, num_steps=3000):
    V = dataset["V"]
    seqs = torch.tensor(dataset["sequences"])
    model = TinyTransformer(V, causal=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(num_steps):
        logits = model(seqs)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            seqs[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def train_bidirectional(dataset, num_steps=5000):
    V = dataset["V"]
    mask_id = dataset["tok2id"]["<MASK>"]
    content_pos = dataset["content_positions"]
    seqs = torch.tensor(dataset["sequences"])
    N = len(seqs)

    model = TinyTransformer(V, causal=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(num_steps):
        masked = seqs.clone()
        mask_info = []
        for b in range(N):
            n_mask = torch.randint(1, len(content_pos) + 1, (1,)).item()
            positions = torch.randperm(len(content_pos))[:n_mask]
            actual_pos = [content_pos[p] for p in positions]
            for pos in actual_pos:
                masked[b, pos] = mask_id
            mask_info.append(actual_pos)

        logits = model(masked)
        loss = 0
        count = 0
        for b in range(N):
            for pos in mask_info[b]:
                loss += F.cross_entropy(logits[b, pos], seqs[b, pos])
                count += 1
        loss = loss / count

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


# ═══════════════════════════════════════════════════════════════════
# Method implementations (all return results + steps_per_sample)
# ═══════════════════════════════════════════════════════════════════

def method_autoregressive(model, dataset, num_samples=500):
    """Standard AR: 1 step per token."""
    V = dataset["V"]
    bos_id = dataset["tok2id"]["<BOS>"]
    seq_len = dataset["seq_len"]
    content_pos = dataset["content_positions"]

    model.eval()
    results = []
    with torch.no_grad():
        for _ in range(num_samples):
            tokens = [bos_id]
            for _ in range(seq_len - 1):
                x = torch.tensor([tokens])
                logits = model(x)
                probs = F.softmax(logits[0, -1], dim=-1)
                tokens.append(torch.multinomial(probs, 1).item())
            results.append(tuple(tokens))

    steps = [len(content_pos)] * num_samples  # one step per content position
    return results, steps


def method_jacobi(model, dataset, num_samples=500, use_sampling=True, max_iters=20):
    """Jacobi iteration: initialize random, refine until convergence."""
    V = dataset["V"]
    bos_id = dataset["tok2id"]["<BOS>"]
    eos_id = dataset["tok2id"]["<EOS>"]
    seq_len = dataset["seq_len"]
    content_pos = dataset["content_positions"]

    # Content tokens to initialize from
    all_content_tokens = set()
    for seq in dataset["sequences"]:
        for pos in content_pos:
            all_content_tokens.add(seq[pos])
    all_content_tokens = list(all_content_tokens)

    model.eval()
    results = []
    steps_list = []

    with torch.no_grad():
        for _ in range(num_samples):
            seq = torch.zeros(1, seq_len, dtype=torch.long)
            seq[0, 0] = bos_id
            seq[0, -1] = eos_id
            for pos in content_pos:
                seq[0, pos] = all_content_tokens[
                    torch.randint(len(all_content_tokens), (1,)).item()]

            for it in range(1, max_iters + 1):
                logits = model(seq)
                if use_sampling:
                    new_tokens = []
                    for i, pos in enumerate(content_pos):
                        probs = F.softmax(logits[0, pos - 1], dim=-1)
                        new_tokens.append(torch.multinomial(probs, 1).item())
                else:
                    new_tokens = [logits[0, pos - 1].argmax().item()
                                  for pos in content_pos]

                new_seq = seq.clone()
                for i, pos in enumerate(content_pos):
                    new_seq[0, pos] = new_tokens[i]

                if torch.equal(new_seq, seq):
                    steps_list.append(it)
                    seq = new_seq
                    break
                seq = new_seq
            else:
                steps_list.append(max_iters)

            results.append(tuple(seq[0].tolist()))

    return results, steps_list


def method_mask_predict(model, dataset, num_samples=500, max_iters=10):
    """Mask-Predict: iteratively unmask most confident positions."""
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

                # Get confidence at masked positions
                conf_pred = []
                for pos in masked:
                    probs = F.softmax(logits[0, pos], dim=-1)
                    max_prob, pred = probs.max(dim=-1)
                    conf_pred.append((pos, max_prob.item(), pred.item()))

                # Sort by confidence, unmask most confident
                conf_pred.sort(key=lambda x: x[1], reverse=True)

                # Unmask at least 1, up to ceil(remaining / remaining_iters)
                remaining_iters = max(1, max_iters - it)
                n_unmask = max(1, len(masked) // remaining_iters)

                for pos, conf, pred in conf_pred[:n_unmask]:
                    seq[0, pos] = pred
                    masked.remove(pos)

            steps_list.append(step_count)
            results.append(tuple(seq[0].tolist()))

    return results, steps_list


def method_mask_predict_adaptive(model, dataset, num_samples=500,
                                  max_iters=10, threshold=0.8):
    """Adaptive Mask-Predict: unmask ALL positions above threshold."""
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

                to_unmask = []
                best = (None, 0, None)
                for pos in masked:
                    probs = F.softmax(logits[0, pos], dim=-1)
                    max_prob, pred = probs.max(dim=-1)
                    conf = max_prob.item()
                    if conf >= threshold:
                        to_unmask.append((pos, pred.item()))
                    if conf > best[1]:
                        best = (pos, conf, pred.item())

                if not to_unmask:
                    to_unmask = [(best[0], best[2])]

                for pos, pred in to_unmask:
                    seq[0, pos] = pred
                    if pos in masked:
                        masked.remove(pos)

            steps_list.append(step_count)
            results.append(tuple(seq[0].tolist()))

    return results, steps_list


def method_speculative(model, dataset, num_samples=500):
    """Speculative decoding: draft random, verify against AR model."""
    V = dataset["V"]
    bos_id = dataset["tok2id"]["<BOS>"]
    eos_id = dataset["tok2id"]["<EOS>"]
    content_pos = dataset["content_positions"]
    seq_len = dataset["seq_len"]

    all_content_tokens = set()
    for seq in dataset["sequences"]:
        for pos in content_pos:
            all_content_tokens.add(seq[pos])
    all_content_tokens = list(all_content_tokens)

    model.eval()
    results = []
    steps_list = []

    with torch.no_grad():
        for _ in range(num_samples):
            tokens = [bos_id]
            passes = 0

            while len(tokens) < seq_len:
                n_draft = seq_len - len(tokens)
                draft = [all_content_tokens[
                    torch.randint(len(all_content_tokens), (1,)).item()]
                    for _ in range(n_draft)]
                candidate = tokens + draft

                x = torch.tensor([candidate])
                logits = model(x)
                passes += 1

                for i in range(len(tokens), seq_len):
                    target_probs = F.softmax(logits[0, i - 1], dim=-1)
                    draft_tok = candidate[i]
                    p_target = target_probs[draft_tok].item()
                    p_draft = 1.0 / len(all_content_tokens)
                    accept_prob = min(1.0, p_target / p_draft)

                    if torch.rand(1).item() < accept_prob:
                        tokens.append(draft_tok)
                    else:
                        adjusted = torch.clamp(target_probs - p_draft, min=0)
                        if adjusted.sum() > 0:
                            adjusted = adjusted / adjusted.sum()
                            tokens.append(torch.multinomial(adjusted, 1).item())
                        else:
                            tokens.append(torch.multinomial(target_probs, 1).item())
                        break

            steps_list.append(passes)
            results.append(tuple(tokens[:seq_len]))

    return results, steps_list


def method_diffusion(dataset, num_samples=500, num_timesteps=10, train_steps=5000):
    """Discrete diffusion with adaptive unmasking."""
    from diffusion import DiffusionTransformer, train_diffusion, diffusion_inference_adaptive

    model = train_diffusion(dataset, num_steps=train_steps, num_timesteps=num_timesteps)
    results, steps = diffusion_inference_adaptive(
        model, dataset, num_samples=num_samples, confidence_threshold=0.8)
    return results, steps, model


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate(results, steps, dataset, method_name):
    """Compute validity, diversity, and step statistics."""
    valid_seqs = [tuple(s) for s in dataset["sequences"]]
    num_samples = len(results)

    # Validity
    valid = sum(1 for r in results if list(r) in dataset["sequences"])

    # Diversity: entropy of output distribution
    counter = Counter(results)
    probs = [c / num_samples for c in counter.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(len(dataset["sequences"]))

    # Steps
    avg_steps = sum(steps) / len(steps)
    min_steps = min(steps)
    max_steps = max(steps)

    return {
        "method": method_name,
        "valid_pct": 100 * valid / num_samples,
        "diversity_entropy": entropy,
        "max_entropy": max_entropy,
        "avg_steps": avg_steps,
        "min_steps": min_steps,
        "max_steps": max_steps,
        "top_sequences": counter.most_common(6),
    }


def print_eval(result, dataset, id2tok):
    print(f"  Method: {result['method']}")
    print(f"  Valid: {result['valid_pct']:.1f}%")
    print(f"  Diversity: {result['diversity_entropy']:.3f} / {result['max_entropy']:.3f} bits")
    print(f"  Steps: avg={result['avg_steps']:.2f}  min={result['min_steps']}  max={result['max_steps']}")
    print(f"  Top sequences:")
    for seq, count in result["top_sequences"]:
        text = " ".join(id2tok[t] for t in seq)
        print(f"    {text:40s}  {count:4d}")
    print()


# ═══════════════════════════════════════════════════════════════════
# Main benchmark
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    NUM_SAMPLES = 200

    # Select key datasets (not all — too verbose)
    test_datasets = [
        dataset_independent,
        dataset_depth1,
        dataset_depth1_wide,
        dataset_depth2,
        dataset_depth3,
        dataset_mixed,
    ]

    all_results = []

    for ds_fn in test_datasets:
        ds = ds_fn()
        print("=" * 70)
        print(f"DATASET: {ds['name']}  (depth={ds['sequential_depth']}, "
              f"{len(ds['sequences'])} seqs, len={ds['seq_len']})")
        print("=" * 70)

        # Train models
        print("Training causal model...")
        causal_model = train_causal(ds, num_steps=2000)
        print("Training bidirectional model...")
        bidir_model = train_bidirectional(ds, num_steps=3000)
        print()

        # Run methods
        methods = [
            ("Autoregressive", lambda: method_autoregressive(causal_model, ds, NUM_SAMPLES)),
            ("Jacobi (argmax)", lambda: method_jacobi(causal_model, ds, NUM_SAMPLES, use_sampling=False)),
            ("Jacobi (sampling)", lambda: method_jacobi(causal_model, ds, NUM_SAMPLES, use_sampling=True)),
            ("Mask-Predict", lambda: method_mask_predict(bidir_model, ds, NUM_SAMPLES)),
            ("Mask-Predict (adaptive)", lambda: method_mask_predict_adaptive(bidir_model, ds, NUM_SAMPLES)),
            ("Speculative", lambda: method_speculative(causal_model, ds, NUM_SAMPLES)),
        ]

        ds_results = []
        for name, fn in methods:
            r, s = fn()
            ev = evaluate(r, s, ds, name)
            print_eval(ev, ds, ds["id2tok"])
            ds_results.append(ev)

        # Diffusion (needs its own training)
        print("Training diffusion model...")
        r, s, _ = method_diffusion(ds, NUM_SAMPLES, num_timesteps=10, train_steps=3000)
        ev = evaluate(r, s, ds, "Diffusion (adaptive)")
        print_eval(ev, ds, ds["id2tok"])
        ds_results.append(ev)

        all_results.append((ds, ds_results))
        print()

    # ── Summary table ──
    print("\n" + "=" * 90)
    print("SUMMARY: Average Steps to Resolve")
    print("=" * 90)
    print(f"{'Dataset':20s} {'Depth':6s} ", end="")
    method_names = ["AR", "Jac-A", "Jac-S", "MP", "MP-Ad", "Spec", "Diff"]
    for m in method_names:
        print(f"{m:>7s}", end="")
    print()
    print("-" * 90)

    for ds, results in all_results:
        depth_str = str(ds["sequential_depth"])
        print(f"{ds['name']:20s} {depth_str:6s} ", end="")
        for ev in results:
            valid = "!" if ev["valid_pct"] < 100 else " "
            print(f"{ev['avg_steps']:6.2f}{valid}", end="")
        print()

    print()
    print("! = <100% validity")
    print()

    print("=" * 90)
    print("SUMMARY: Validity %")
    print("=" * 90)
    print(f"{'Dataset':20s} {'Depth':6s} ", end="")
    for m in method_names:
        print(f"{m:>7s}", end="")
    print()
    print("-" * 90)

    for ds, results in all_results:
        depth_str = str(ds["sequential_depth"])
        print(f"{ds['name']:20s} {depth_str:6s} ", end="")
        for ev in results:
            print(f"{ev['valid_pct']:6.1f}%", end="")
        print()

    print()
    print("=" * 90)
    print("SUMMARY: Diversity (bits of entropy)")
    print("=" * 90)
    print(f"{'Dataset':20s} {'Max':6s} ", end="")
    for m in method_names:
        print(f"{m:>7s}", end="")
    print()
    print("-" * 90)

    for ds, results in all_results:
        max_ent = f"{results[0]['max_entropy']:.2f}"
        print(f"{ds['name']:20s} {max_ent:6s} ", end="")
        for ev in results:
            print(f"{ev['diversity_entropy']:6.3f} ", end="")
        print()

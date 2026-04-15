"""
Advanced Discrete Diffusion Methods for Parallel Generation

Implements three approaches based on recent papers:

1. MDLM (Sahoo et al., 2024): Absorbing-state diffusion with proper ELBO
   loss weighting and sampling-based inference.

2. Uniform Diffusion (Gemini Diffusion style): Tokens can transition to
   ANY state (not just MASK), enabling self-correction at each step.

3. Planned Denoising (DDPD-inspired): Separate planning of which positions
   to denoise first, based on estimated corruption level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter
from common import TinyTransformer


# ═══════════════════════════════════════════════════════════════════
# Shared: Timestep-conditioned bidirectional transformer
# ═══════════════════════════════════════════════════════════════════

class DiffusionTransformerV2(nn.Module):
    """Bidirectional transformer with timestep conditioning."""
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2,
                 max_len=16, num_timesteps=100):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.0, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, t_continuous):
        """
        x: (B, T) token ids
        t_continuous: (B,) float in [0, 1]
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        t_emb = self.time_mlp(t_continuous.unsqueeze(-1))  # (B, d_model)
        h = self.tok_emb(x) + self.pos_emb(pos) + t_emb.unsqueeze(1)
        h = self.transformer(h)
        return self.head(h)


# ═══════════════════════════════════════════════════════════════════
# Method 1: MDLM — Absorbing state with ELBO loss weighting
# ═══════════════════════════════════════════════════════════════════

def noise_schedule_cosine(t):
    """Cosine noise schedule: alpha(t) = cos(pi*t/2)^2"""
    return torch.cos(t * math.pi / 2) ** 2


def train_mdlm(dataset, num_steps=5000, num_timesteps=100):
    """
    MDLM training (Sahoo et al., 2024):
    - Absorbing state (MASK) forward process
    - Cosine noise schedule
    - ELBO-derived loss with alpha'(t) / (1 - alpha(t)) weighting
    - Low-discrepancy timestep sampling
    """
    V = dataset["V"]
    mask_id = dataset["tok2id"]["<MASK>"]
    content_pos = dataset["content_positions"]
    seqs = torch.tensor(dataset["sequences"])
    N = len(seqs)

    model = DiffusionTransformerV2(V, num_timesteps=num_timesteps)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(num_steps):
        idx = torch.randint(N, (min(N, 16),))
        clean = seqs[idx].clone()
        B = clean.shape[0]

        # Low-discrepancy timestep sampling (stratified)
        u = (torch.arange(B, dtype=torch.float32) + torch.rand(B)) / B
        t = u.clamp(1e-4, 1 - 1e-4)  # avoid boundaries

        # Compute alpha(t) via cosine schedule
        alpha_t = noise_schedule_cosine(t)  # (B,)

        # Forward process: mask each content token independently with prob 1 - alpha(t)
        noisy = clean.clone()
        for b in range(B):
            mask_prob = 1 - alpha_t[b].item()
            for pos in content_pos:
                if torch.rand(1).item() < mask_prob:
                    noisy[b, pos] = mask_id

        # Forward pass
        logits = model(noisy, t)  # (B, L, V)

        # ELBO loss with weighting: -alpha'(t) / (1 - alpha(t))
        # For cosine: alpha'(t) = -pi * sin(pi*t/2) * cos(pi*t/2)
        # Weight = pi * sin(pi*t/2) * cos(pi*t/2) / (1 - cos^2(pi*t/2))
        #        = pi * cos(pi*t/2) / sin(pi*t/2)  = pi / tan(pi*t/2)
        alpha_prime = -math.pi * torch.sin(t * math.pi / 2) * torch.cos(t * math.pi / 2)
        weight = (-alpha_prime / (1 - alpha_t + 1e-8)).clamp(max=10.0)  # (B,)

        # Loss: weighted cross-entropy on content positions
        loss = 0
        count = 0
        for b in range(B):
            for pos in content_pos:
                ce = F.cross_entropy(logits[b, pos], clean[b, pos], reduction='none')
                loss += weight[b] * ce
                count += 1
        loss = loss / count

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1:5d}  loss={loss.item():.4f}")

    return model


def inference_mdlm(model, dataset, num_samples=200, num_steps=10):
    """
    MDLM inference: ancestral sampling with proper noise schedule.
    At each step t (from 1 to 0):
    - For masked positions, sample from predicted distribution
    - Carry over unmasked positions unchanged
    - Randomly re-mask some predictions based on schedule
    """
    V = dataset["V"]
    mask_id = dataset["tok2id"]["<MASK>"]
    bos_id = dataset["tok2id"]["<BOS>"]
    eos_id = dataset["tok2id"]["<EOS>"]
    content_pos = dataset["content_positions"]
    seq_len = dataset["seq_len"]

    model.eval()
    results = []
    steps_used = []

    with torch.no_grad():
        for _ in range(num_samples):
            # Start fully masked
            seq = torch.zeros(1, seq_len, dtype=torch.long)
            seq[0, 0] = bos_id
            seq[0, -1] = eos_id
            for pos in content_pos:
                seq[0, pos] = mask_id

            step_count = 0
            timesteps = torch.linspace(1, 0, num_steps + 1)

            for i in range(num_steps):
                t_now = timesteps[i]
                t_next = timesteps[i + 1]
                alpha_now = noise_schedule_cosine(t_now)
                alpha_next = noise_schedule_cosine(t_next)

                t_input = torch.tensor([t_now.item()])
                logits = model(seq, t_input)
                step_count += 1

                for pos in content_pos:
                    if seq[0, pos] == mask_id:
                        probs = F.softmax(logits[0, pos], dim=-1)
                        # Zero out MASK token probability
                        probs[mask_id] = 0
                        if probs.sum() > 0:
                            probs = probs / probs.sum()
                        sampled = torch.multinomial(probs, 1).item()

                        # Decide whether to keep or re-mask
                        # Prob of staying unmasked: alpha_next / alpha_now
                        keep_prob = (alpha_next / (alpha_now + 1e-8)).item()
                        keep_prob = min(keep_prob, 1.0)

                        if i == num_steps - 1:
                            # Last step: always keep
                            seq[0, pos] = sampled
                        elif torch.rand(1).item() < keep_prob:
                            seq[0, pos] = sampled
                        # else: stays masked for next iteration

            results.append(tuple(seq[0].tolist()))
            steps_used.append(step_count)

    return results, steps_used


# ═══════════════════════════════════════════════════════════════════
# Method 2: Uniform Diffusion (Gemini Diffusion style)
# ═══════════════════════════════════════════════════════════════════

def train_uniform_diffusion(dataset, num_steps=5000):
    """
    Uniform diffusion: tokens can transition to ANY token (not just MASK).
    Forward process: with probability (1-alpha(t)), replace token with
    uniform random token from vocabulary.

    This enables self-correction: at inference, ALL positions are updated
    at each step, not just masked ones. Tokens can change from one wrong
    value to the correct one.
    """
    V = dataset["V"]
    mask_id = dataset["tok2id"]["<MASK>"]
    content_pos = dataset["content_positions"]
    seqs = torch.tensor(dataset["sequences"])
    N = len(seqs)

    # Content token ids (exclude BOS, EOS, MASK)
    content_token_ids = [i for i in range(V) if i not in
                         [dataset["tok2id"]["<BOS>"], dataset["tok2id"]["<EOS>"],
                          dataset["tok2id"]["<MASK>"]]]

    model = DiffusionTransformerV2(V)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(num_steps):
        idx = torch.randint(N, (min(N, 16),))
        clean = seqs[idx].clone()
        B = clean.shape[0]

        # Random timesteps
        t = torch.rand(B).clamp(1e-4, 1 - 1e-4)
        alpha_t = noise_schedule_cosine(t)

        # Forward: replace tokens with uniform random with prob 1-alpha(t)
        noisy = clean.clone()
        for b in range(B):
            corrupt_prob = 1 - alpha_t[b].item()
            for pos in content_pos:
                if torch.rand(1).item() < corrupt_prob:
                    # Replace with random content token (NOT mask)
                    noisy[b, pos] = content_token_ids[
                        torch.randint(len(content_token_ids), (1,)).item()]

        logits = model(noisy, t)

        # Simple cross-entropy loss (predict clean from noisy)
        loss = 0
        count = 0
        for pos in content_pos:
            loss += F.cross_entropy(logits[:, pos], clean[:, pos])
            count += 1
        loss = loss / count

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1:5d}  loss={loss.item():.4f}")

    return model


def inference_uniform(model, dataset, num_samples=200, num_steps=10):
    """
    Uniform diffusion inference (Gemini-style):
    Start with random tokens, iteratively refine ALL positions.
    At each step, the model predicts the clean token for every position
    and we sample, allowing self-correction.
    """
    V = dataset["V"]
    bos_id = dataset["tok2id"]["<BOS>"]
    eos_id = dataset["tok2id"]["<EOS>"]
    mask_id = dataset["tok2id"]["<MASK>"]
    content_pos = dataset["content_positions"]
    seq_len = dataset["seq_len"]

    content_token_ids = [i for i in range(V) if i not in [bos_id, eos_id, mask_id]]

    model.eval()
    results = []
    steps_used = []

    with torch.no_grad():
        for _ in range(num_samples):
            # Start with random tokens
            seq = torch.zeros(1, seq_len, dtype=torch.long)
            seq[0, 0] = bos_id
            seq[0, -1] = eos_id
            for pos in content_pos:
                seq[0, pos] = content_token_ids[
                    torch.randint(len(content_token_ids), (1,)).item()]

            timesteps = torch.linspace(1, 0, num_steps + 1)

            for i in range(num_steps):
                t_now = timesteps[i]
                t_input = torch.tensor([t_now.item()])
                logits = model(seq, t_input)

                # Update ALL content positions (self-correction)
                for pos in content_pos:
                    probs = F.softmax(logits[0, pos], dim=-1)
                    # Zero out special tokens
                    probs[bos_id] = 0
                    probs[eos_id] = 0
                    probs[mask_id] = 0
                    if probs.sum() > 0:
                        probs = probs / probs.sum()

                    if i == num_steps - 1:
                        # Last step: argmax for stability
                        seq[0, pos] = probs.argmax()
                    else:
                        # Mix: with probability proportional to t, add noise
                        # Lower t = more confident = less noise
                        temperature = 0.5 + t_now.item()  # 0.5 at t=0, 1.5 at t=1
                        tempered = (probs.log() / temperature).softmax(dim=-1)
                        seq[0, pos] = torch.multinomial(tempered, 1).item()

            results.append(tuple(seq[0].tolist()))
            steps_used.append(num_steps)

    return results, steps_used


# ═══════════════════════════════════════════════════════════════════
# Method 3: Planned Denoising (DDPD-inspired)
# ═══════════════════════════════════════════════════════════════════

def train_planned_denoising(dataset, num_steps=5000):
    """
    DDPD-inspired: Train a model that can both predict clean tokens
    AND estimate which positions are most corrupted (planning).

    Uses absorbing-state diffusion but adds a corruption predictor head.
    """
    V = dataset["V"]
    mask_id = dataset["tok2id"]["<MASK>"]
    content_pos = dataset["content_positions"]
    seqs = torch.tensor(dataset["sequences"])
    N = len(seqs)

    model = DiffusionTransformerV2(V)
    # Extra head: predict whether each position is corrupted (binary)
    corruption_head = nn.Linear(64, 1)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(corruption_head.parameters()), lr=1e-3)

    for step in range(num_steps):
        idx = torch.randint(N, (min(N, 16),))
        clean = seqs[idx].clone()
        B = clean.shape[0]

        t = torch.rand(B).clamp(1e-4, 1 - 1e-4)
        alpha_t = noise_schedule_cosine(t)

        noisy = clean.clone()
        is_corrupted = torch.zeros(B, clean.shape[1])
        for b in range(B):
            mask_prob = 1 - alpha_t[b].item()
            for pos in content_pos:
                if torch.rand(1).item() < mask_prob:
                    noisy[b, pos] = mask_id
                    is_corrupted[b, pos] = 1.0

        # Get transformer hidden states
        T_len = noisy.shape[1]
        pos_ids = torch.arange(T_len, device=noisy.device).unsqueeze(0)
        h = model.tok_emb(noisy) + model.pos_emb(pos_ids) + \
            model.time_mlp(t.unsqueeze(-1)).unsqueeze(1)
        h = model.transformer(h)

        # Token prediction loss
        logits = model.head(h)
        token_loss = 0
        count = 0
        for pos in content_pos:
            token_loss += F.cross_entropy(logits[:, pos], clean[:, pos])
            count += 1
        token_loss = token_loss / count

        # Corruption prediction loss (binary cross-entropy)
        corruption_logits = corruption_head(h).squeeze(-1)  # (B, L)
        corruption_loss = 0
        for pos in content_pos:
            corruption_loss += F.binary_cross_entropy_with_logits(
                corruption_logits[:, pos], is_corrupted[:, pos])
        corruption_loss = corruption_loss / len(content_pos)

        loss = token_loss + 0.5 * corruption_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 1000 == 0:
            print(f"  Step {step+1:5d}  loss={loss.item():.4f}  "
                  f"(tok={token_loss.item():.4f} corr={corruption_loss.item():.4f})")

    return model, corruption_head


def inference_planned(model, corruption_head, dataset, num_samples=200, max_steps=10):
    """
    Planned denoising inference:
    1. Start fully masked
    2. At each step, predict tokens AND estimate corruption
    3. Unmask positions the model is MOST CONFIDENT are correct
       (lowest predicted corruption score)
    4. Sample (not argmax) at uncertain positions for diversity
    """
    V = dataset["V"]
    mask_id = dataset["tok2id"]["<MASK>"]
    bos_id = dataset["tok2id"]["<BOS>"]
    eos_id = dataset["tok2id"]["<EOS>"]
    content_pos = dataset["content_positions"]
    seq_len = dataset["seq_len"]

    model.eval()
    corruption_head.eval()
    results = []
    steps_used = []

    with torch.no_grad():
        for _ in range(num_samples):
            seq = torch.zeros(1, seq_len, dtype=torch.long)
            seq[0, 0] = bos_id
            seq[0, -1] = eos_id
            for pos in content_pos:
                seq[0, pos] = mask_id

            masked = list(content_pos)
            step_count = 0

            for it in range(max_steps):
                if not masked:
                    break

                # Estimate timestep from fraction of masked positions
                t_est = len(masked) / len(content_pos)
                t_input = torch.tensor([t_est])

                # Get hidden states
                pos_ids = torch.arange(seq_len).unsqueeze(0)
                h = model.tok_emb(seq) + model.pos_emb(pos_ids) + \
                    model.time_mlp(t_input.unsqueeze(-1)).unsqueeze(1)
                h = model.transformer(h)

                logits = model.head(h)
                corruption_scores = torch.sigmoid(
                    corruption_head(h).squeeze(-1))  # (1, L)
                step_count += 1

                # For each masked position: get prediction + corruption score
                pos_info = []
                for pos in masked:
                    probs = F.softmax(logits[0, pos], dim=-1)
                    probs[mask_id] = 0
                    if probs.sum() > 0:
                        probs = probs / probs.sum()
                    corr = corruption_scores[0, pos].item()
                    max_prob = probs.max().item()
                    pos_info.append({
                        "pos": pos, "probs": probs,
                        "corruption": corr, "confidence": max_prob,
                    })

                # Sort by confidence (high) and low corruption → commit first
                # Combined score: confidence * (1 - corruption)
                pos_info.sort(key=lambda x: x["confidence"] * (1 - x["corruption"]),
                              reverse=True)

                # Unmask the most confident positions
                n_unmask = max(1, len(masked) // max(1, max_steps - it))
                if it == max_steps - 1:
                    n_unmask = len(masked)

                for info in pos_info[:n_unmask]:
                    # Sample from distribution (for diversity)
                    tok = torch.multinomial(info["probs"], 1).item()
                    seq[0, info["pos"]] = tok
                    masked.remove(info["pos"])

            results.append(tuple(seq[0].tolist()))
            steps_used.append(step_count)

    return results, steps_used


# ═══════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import math
    from datasets_v3 import ALL_V3_DATASETS
    from benchmark import method_autoregressive, train_causal
    from entropy_aware import method_sampling_mask_predict, method_bidir_speculative
    from benchmark import train_bidirectional

    torch.manual_seed(42)
    NUM = 200

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
            "entropy": entropy,
            "max_entropy": max_entropy,
            "e_ratio": entropy / max_entropy if max_entropy > 0 else 0,
            "avg_steps": avg_steps,
            "step_ratio": avg_steps / dataset["min_steps"] if dataset["min_steps"] > 0 else avg_steps,
        }

    # Select key datasets
    test_datasets = [
        ("Wide tree (d=1)", ALL_V3_DATASETS[2][1]),
        ("Diamond (d=2)", ALL_V3_DATASETS[3][1]),
        ("Hourglass (d=2)", ALL_V3_DATASETS[6][1]),
        ("Deep narrow (d=4)", ALL_V3_DATASETS[8][1]),
    ]

    methods_order = ["AR", "Samp-MP", "Bidir-Spec", "MDLM", "Uniform", "Planned"]

    all_results = []

    for label, ds_fn in test_datasets:
        ds = ds_fn()
        print(f"\n{'='*70}")
        print(f"  {label}  |  seqs={len(ds['sequences'])}  min_steps={ds['min_steps']}")
        print(f"{'='*70}")

        # Train baselines
        print("  Training causal...", end=" ", flush=True)
        causal = train_causal(ds, num_steps=3000)
        print("bidir...", end=" ", flush=True)
        bidir = train_bidirectional(ds, num_steps=5000)
        print("done")

        ds_results = {}

        # Baselines
        r, s = method_autoregressive(causal, ds, NUM)
        ds_results["AR"] = evaluate(r, s, ds, "AR")

        r, s = method_sampling_mask_predict(bidir, ds, NUM)
        ds_results["Samp-MP"] = evaluate(r, s, ds, "Samp-MP")

        r, s = method_bidir_speculative(causal, bidir, ds, NUM)
        ds_results["Bidir-Spec"] = evaluate(r, s, ds, "Bidir-Spec")

        # MDLM
        print("  Training MDLM...", end=" ", flush=True)
        mdlm = train_mdlm(ds, num_steps=5000)
        print("done")
        for n_steps in [5, 10]:
            r, s = inference_mdlm(mdlm, ds, NUM, num_steps=n_steps)
            ev = evaluate(r, s, ds, f"MDLM-{n_steps}")
            if n_steps == 10:
                ds_results["MDLM"] = ev

        # Uniform Diffusion
        print("  Training Uniform...", end=" ", flush=True)
        uni = train_uniform_diffusion(ds, num_steps=5000)
        print("done")
        for n_steps in [5, 10]:
            r, s = inference_uniform(uni, ds, NUM, num_steps=n_steps)
            ev = evaluate(r, s, ds, f"Uniform-{n_steps}")
            if n_steps == 10:
                ds_results["Uniform"] = ev

        # Planned Denoising
        print("  Training Planned...", end=" ", flush=True)
        plan_model, corr_head = train_planned_denoising(ds, num_steps=5000)
        print("done")
        r, s = inference_planned(plan_model, corr_head, ds, NUM, max_steps=10)
        ds_results["Planned"] = evaluate(r, s, ds, "Planned")

        # Print
        print(f"\n  {'Method':12s} {'Steps':>7s} {'Ratio':>7s} {'Valid%':>7s} "
              f"{'Entropy':>8s} {'E-ratio':>8s}")
        print(f"  {'-'*53}")
        for m in methods_order:
            if m in ds_results:
                ev = ds_results[m]
                print(f"  {m:12s} {ev['avg_steps']:6.2f} {ev['step_ratio']:6.2f}x "
                      f"{ev['valid_pct']:5.1f}% {ev['entropy']:7.3f} "
                      f"{ev['e_ratio']:7.1%}")

        all_results.append((label, ds, ds_results))

    # Summary
    print(f"\n\n{'='*90}")
    print("SUMMARY: All Methods Compared")
    print(f"{'='*90}")
    print(f"{'Dataset':22s} {'min':>4s}", end="")
    for m in methods_order:
        print(f" {m:>12s}", end="")
    print()
    print(f"{'':22s} {'':>4s}", end="")
    for m in methods_order:
        print(f" {'stp  v% e%':>12s}", end="")
    print()
    print("-" * (27 + 13 * len(methods_order)))

    for label, ds, results in all_results:
        min_s = ds["min_steps"]
        print(f"{label:22s} {min_s:>4d}", end="")
        for m in methods_order:
            if m in results:
                ev = results[m]
                v_mark = " " if ev["valid_pct"] >= 99 else "!"
                print(f" {ev['avg_steps']:4.1f} {ev['valid_pct']:3.0f}{v_mark}"
                      f"{ev['e_ratio']:3.0%}", end="")
            else:
                print(f" {'---':>12s}", end="")
        print()

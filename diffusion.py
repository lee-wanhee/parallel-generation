"""
Discrete Diffusion Generation (Absorbing State)

Paper: "Structured Denoising Diffusion Models in Discrete State-Spaces" (Austin et al., 2021)
       "Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning" (Chen et al., 2023)

Core idea (absorbing state diffusion):
- Forward process: gradually replace tokens with <MASK> (absorbing state)
- Reverse process: predict masked tokens from partially masked sequences
- Training: at each noise level t, mask ~t fraction of tokens, predict originals
- Inference: start fully masked, iteratively unmask by predicting and keeping
  highest-confidence tokens

Each denoising step = one step of uncertainty removal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


class DiffusionTransformer(nn.Module):
    """Bidirectional transformer conditioned on timestep for discrete diffusion."""
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2,
                 max_len=16, num_timesteps=10):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.time_emb = nn.Embedding(num_timesteps + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.0, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.num_timesteps = num_timesteps

    def forward(self, x, t):
        """
        x: (B, T) token ids (may contain MASK tokens)
        t: (B,) integer timestep
        Returns: (B, T, V) logits predicting original tokens
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos) + self.time_emb(t).unsqueeze(1)
        h = self.transformer(h)  # bidirectional
        return self.head(h)


def train_diffusion(dataset, num_steps=5000, num_timesteps=10):
    """
    Train a discrete diffusion model with absorbing state (MASK).

    Forward process: at timestep t (1..T), each content token is independently
    replaced by MASK with probability t/T.
    """
    V = dataset["V"]
    mask_id = dataset["tok2id"]["<MASK>"]
    bos_id = dataset["tok2id"]["<BOS>"]
    eos_id = dataset["tok2id"]["<EOS>"]
    content_pos = dataset["content_positions"]
    seqs = torch.tensor(dataset["sequences"])  # (N, L)
    N = len(seqs)

    model = DiffusionTransformer(V, num_timesteps=num_timesteps)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(num_steps):
        # Sample random sequences and timesteps
        idx = torch.randint(N, (min(N, 8),))
        batch = seqs[idx].clone()  # (B, L)
        B = batch.shape[0]

        # Random timestep for each sample
        t = torch.randint(1, num_timesteps + 1, (B,))

        # Apply noise: mask content positions with probability t/T
        clean = batch.clone()
        for b in range(B):
            mask_prob = t[b].float() / num_timesteps
            for pos in content_pos:
                if torch.rand(1).item() < mask_prob:
                    batch[b, pos] = mask_id

        logits = model(batch, t)  # (B, L, V)

        # Loss only on content positions (predict original tokens)
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


def diffusion_inference(model, dataset, num_samples=1000, schedule="linear"):
    """
    Discrete diffusion inference:
    1. Start: all content positions masked
    2. At each step t (from T to 1):
       - Feed current sequence + timestep t to model
       - Get predictions for all masked positions
       - Unmask positions with highest confidence
       - Number to unmask follows a schedule

    Returns (results, steps_per_sample)
    """
    mask_id = dataset["tok2id"]["<MASK>"]
    bos_id = dataset["tok2id"]["<BOS>"]
    eos_id = dataset["tok2id"]["<EOS>"]
    content_pos = dataset["content_positions"]
    seq_len = dataset["seq_len"]
    num_timesteps = model.num_timesteps
    id2tok = dataset["id2tok"]

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

            masked_positions = list(content_pos)
            n_content = len(content_pos)
            step_count = 0

            for t_idx in range(num_timesteps, 0, -1):
                if not masked_positions:
                    break

                t = torch.tensor([t_idx])
                logits = model(seq, t)
                step_count += 1

                # Get confidence at masked positions
                confidences = {}
                predictions = {}
                for pos in masked_positions:
                    probs = F.softmax(logits[0, pos], dim=-1)
                    max_prob, pred = probs.max(dim=-1)
                    confidences[pos] = max_prob.item()
                    predictions[pos] = pred.item()

                # Schedule: how many to unmask at this step
                remaining_steps = t_idx
                n_to_unmask = max(1, len(masked_positions) // remaining_steps)
                if t_idx == 1:
                    n_to_unmask = len(masked_positions)

                # Unmask most confident
                sorted_pos = sorted(masked_positions,
                                    key=lambda p: confidences[p], reverse=True)
                for pos in sorted_pos[:n_to_unmask]:
                    seq[0, pos] = predictions[pos]
                    masked_positions.remove(pos)

            results.append(tuple(seq[0].tolist()))
            steps_used.append(step_count)

    return results, steps_used


def diffusion_inference_adaptive(model, dataset, num_samples=1000,
                                  confidence_threshold=0.9):
    """
    Adaptive diffusion: unmask ALL positions above confidence threshold
    at each step. Stops as soon as all positions are unmasked.

    This tests whether the model can adaptively resolve easy samples
    in fewer steps.
    """
    mask_id = dataset["tok2id"]["<MASK>"]
    bos_id = dataset["tok2id"]["<BOS>"]
    eos_id = dataset["tok2id"]["<EOS>"]
    content_pos = dataset["content_positions"]
    seq_len = dataset["seq_len"]
    num_timesteps = model.num_timesteps

    model.eval()
    results = []
    steps_used = []

    with torch.no_grad():
        for _ in range(num_samples):
            seq = torch.zeros(1, seq_len, dtype=torch.long)
            seq[0, 0] = bos_id
            seq[0, -1] = eos_id
            for pos in content_pos:
                seq[0, pos] = mask_id

            masked_positions = list(content_pos)
            step_count = 0

            for t_idx in range(num_timesteps, 0, -1):
                if not masked_positions:
                    break

                t = torch.tensor([t_idx])
                logits = model(seq, t)
                step_count += 1

                # Unmask everything above threshold, or at least 1
                to_unmask = []
                best_pos, best_conf, best_pred = None, 0, None

                for pos in masked_positions:
                    probs = F.softmax(logits[0, pos], dim=-1)
                    max_prob, pred = probs.max(dim=-1)
                    conf = max_prob.item()

                    if conf >= confidence_threshold:
                        to_unmask.append((pos, pred.item()))

                    if conf > best_conf:
                        best_pos, best_conf, best_pred = pos, conf, pred.item()

                # If nothing above threshold, unmask the most confident one
                if not to_unmask:
                    to_unmask = [(best_pos, best_pred)]

                for pos, pred in to_unmask:
                    seq[0, pos] = pred
                    if pos in masked_positions:
                        masked_positions.remove(pos)

            results.append(tuple(seq[0].tolist()))
            steps_used.append(step_count)

    return results, steps_used


if __name__ == "__main__":
    from datasets import dataset_depth1, dataset_depth2, dataset_mixed

    torch.manual_seed(42)

    for ds_fn in [dataset_depth1, dataset_depth2, dataset_mixed]:
        ds = ds_fn()
        print("=" * 60)
        print(f"DIFFUSION on {ds['name']} (depth={ds['sequential_depth']})")
        print("=" * 60)

        print("Training...")
        model = train_diffusion(ds, num_steps=5000)
        print()

        print("── Fixed schedule ──")
        results, steps = diffusion_inference(model, ds, num_samples=500)
        valid = 0
        counter = Counter()
        for seq in results:
            text = " ".join(ds["id2tok"][t] for t in seq)
            counter[text] += 1
            if list(seq) in ds["sequences"]:
                valid += 1
        print(f"Valid: {valid}/{len(results)} ({100*valid/len(results):.1f}%)")
        print(f"Avg steps: {sum(steps)/len(steps):.2f}")
        for seq, count in counter.most_common(8):
            print(f"  {seq:40s}  {count:4d}  ({100*count/len(results):.1f}%)")
        print()

        print("── Adaptive (threshold=0.9) ──")
        results, steps = diffusion_inference_adaptive(model, ds, num_samples=500)
        valid = 0
        counter = Counter()
        for seq in results:
            text = " ".join(ds["id2tok"][t] for t in seq)
            counter[text] += 1
            if list(seq) in ds["sequences"]:
                valid += 1
        print(f"Valid: {valid}/{len(results)} ({100*valid/len(results):.1f}%)")
        print(f"Avg steps: {sum(steps)/len(steps):.2f}")
        for seq, count in counter.most_common(8):
            print(f"  {seq:40s}  {count:4d}  ({100*count/len(results):.1f}%)")
        print()

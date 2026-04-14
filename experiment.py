"""
Experiment: Autoregressive vs Parallel Generation

Training data: Two sequences with equal probability:
  [BOS] I am [EOS]
  [BOS] We are [EOS]

Autoregressive inference correctly produces "I am" or "We are".
Parallel inference produces wrong combos like "I are" / "We am"
because all positions are predicted from BOS independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

# ── Vocab ──
VOCAB = ["<BOS>", "<EOS>", "I", "am", "We", "are"]
tok2id = {t: i for i, t in enumerate(VOCAB)}
id2tok = {i: t for t, i in tok2id.items()}
V = len(VOCAB)

# ── Training sequences ──
seq1 = [tok2id["<BOS>"], tok2id["I"], tok2id["am"], tok2id["<EOS>"]]
seq2 = [tok2id["<BOS>"], tok2id["We"], tok2id["are"], tok2id["<EOS>"]]
SEQ_LEN = 4


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, max_len=16):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.0, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.transformer(h, mask=mask, is_causal=True)
        return self.head(h)


def train(num_steps=2000):
    model = TinyTransformer(V)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = torch.tensor([seq1, seq2])

    print("=" * 50)
    print("TRAINING")
    print("=" * 50)
    print(f"Sequence 1: {[id2tok[i] for i in seq1]}")
    print(f"Sequence 2: {[id2tok[i] for i in seq2]}")
    print()

    for step in range(num_steps):
        logits = model(data)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            data[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 500 == 0:
            print(f"Step {step+1:4d}  loss={loss.item():.4f}")

    print()
    return model


def autoregressive_inference(model, num_samples=1000):
    """Standard autoregressive: sample one token at a time, feed it back."""
    print("=" * 50)
    print("AUTOREGRESSIVE INFERENCE")
    print("=" * 50)

    model.eval()
    results = []
    with torch.no_grad():
        for _ in range(num_samples):
            tokens = [tok2id["<BOS>"]]
            for _ in range(SEQ_LEN - 1):
                x = torch.tensor([tokens])
                logits = model(x)
                probs = F.softmax(logits[0, -1], dim=-1)
                tokens.append(torch.multinomial(probs, 1).item())
            results.append(tuple(tokens))

    _print_results(results, num_samples)


def parallel_inference(model, num_samples=1000):
    """
    Parallel: feed [BOS, BOS, BOS, BOS] so each position only sees BOS
    (via causal mask), then sample all positions independently.
    """
    print("=" * 50)
    print("PARALLEL INFERENCE")
    print("=" * 50)
    print("Input: [BOS, BOS, BOS, BOS] — each position attends only to BOS,")
    print("then positions 1, 2, 3 are sampled independently.\n")

    model.eval()
    results = []
    with torch.no_grad():
        bos_seq = torch.tensor([[tok2id["<BOS>"]] * SEQ_LEN])
        logits = model(bos_seq)
        probs = [F.softmax(logits[0, i], dim=-1) for i in range(SEQ_LEN - 1)]

        for i, p in enumerate(probs):
            topk = torch.topk(p, 3)
            entries = "  ".join(f"{id2tok[idx.item()]}={prob.item():.3f}"
                                for prob, idx in zip(topk.values, topk.indices))
            print(f"Position {i+1} top probs: {entries}")
        print()

        for _ in range(num_samples):
            sampled = [torch.multinomial(p, 1).item() for p in probs]
            results.append((tok2id["<BOS>"], *sampled))

    _print_results(results, num_samples)


def _print_results(results, num_samples):
    counter = Counter()
    for seq in results:
        counter[" ".join(id2tok[t] for t in seq)] += 1

    print(f"Results over {num_samples} samples:")
    for seq, count in counter.most_common():
        print(f"  {seq:30s}  {count:4d}  ({100*count/num_samples:.1f}%)")
    print()


if __name__ == "__main__":
    torch.manual_seed(42)
    model = train()
    autoregressive_inference(model, num_samples=1000)
    parallel_inference(model, num_samples=1000)

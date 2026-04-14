"""Shared vocab, data, model, and utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

# ── Vocab ──
VOCAB = ["<BOS>", "<EOS>", "<MASK>", "I", "am", "We", "are"]
tok2id = {t: i for i, t in enumerate(VOCAB)}
id2tok = {i: t for t, i in tok2id.items()}
V = len(VOCAB)
BOS = tok2id["<BOS>"]
EOS = tok2id["<EOS>"]
MASK = tok2id["<MASK>"]

# ── Training sequences ──
seq1 = [BOS, tok2id["I"], tok2id["am"], EOS]
seq2 = [BOS, tok2id["We"], tok2id["are"], EOS]
SEQ_LEN = 4


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, max_len=16,
                 causal=True):
        super().__init__()
        self.causal = causal
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
        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            h = self.transformer(h, mask=mask, is_causal=True)
        else:
            h = self.transformer(h)
        return self.head(h)


def train_model(causal=True, num_steps=2000, lr=1e-3):
    model = TinyTransformer(V, causal=causal)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data = torch.tensor([seq1, seq2])

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
            print(f"  Step {step+1:4d}  loss={loss.item():.4f}")

    return model


def print_results(results, num_samples):
    counter = Counter()
    for seq in results:
        counter[" ".join(id2tok[t] for t in seq)] += 1

    valid = 0
    for seq, count in counter.most_common():
        tokens = seq.split()
        if tokens == ["<BOS>", "I", "am", "<EOS>"] or tokens == ["<BOS>", "We", "are", "<EOS>"]:
            valid += count
        print(f"  {seq:30s}  {count:4d}  ({100*count/num_samples:.1f}%)")
    print(f"\n  Valid sequences: {valid}/{num_samples} ({100*valid/num_samples:.1f}%)")
    print()
    return valid / num_samples

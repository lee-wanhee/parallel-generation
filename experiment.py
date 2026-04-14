"""
Experiment: Autoregressive vs Naive Parallel Generation (baseline)

Training data: Two sequences with equal probability:
  [BOS] I am [EOS]
  [BOS] We are [EOS]

Autoregressive inference correctly produces "I am" or "We are".
Parallel inference produces wrong combos because all positions are
predicted from BOS independently.
"""

import torch
import torch.nn.functional as F
from common import (V, BOS, SEQ_LEN, id2tok, tok2id, train_model, print_results)


def autoregressive_inference(model, num_samples=1000):
    print("=" * 60)
    print("AUTOREGRESSIVE INFERENCE")
    print("=" * 60)

    model.eval()
    results = []
    with torch.no_grad():
        for _ in range(num_samples):
            tokens = [BOS]
            for _ in range(SEQ_LEN - 1):
                x = torch.tensor([tokens])
                logits = model(x)
                probs = F.softmax(logits[0, -1], dim=-1)
                tokens.append(torch.multinomial(probs, 1).item())
            results.append(tuple(tokens))

    print_results(results, num_samples)


def parallel_inference(model, num_samples=1000):
    print("=" * 60)
    print("NAIVE PARALLEL INFERENCE")
    print("=" * 60)
    print("Input: [BOS, BOS, BOS, BOS] — each position attends only to BOS,")
    print("then positions 1, 2, 3 are sampled independently.\n")

    model.eval()
    results = []
    with torch.no_grad():
        bos_seq = torch.tensor([[BOS] * SEQ_LEN])
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
            results.append((BOS, *sampled))

    print_results(results, num_samples)


if __name__ == "__main__":
    torch.manual_seed(42)

    print("Training causal model...")
    model = train_model(causal=True)
    print()

    autoregressive_inference(model, num_samples=1000)
    parallel_inference(model, num_samples=1000)

"""
Jacobi Decoding: Fixed-point iteration on an autoregressive model.

Paper: "Lookahead Decoding" / "Consistency LLMs" lineage.
Core idea: Initialize all positions with random tokens, then iteratively
run the causal model and replace each position with its argmax prediction.
Repeat until the sequence stops changing (fixed point).

This uses the SAME causal model trained for autoregressive generation —
no retraining needed.
"""

import torch
import torch.nn.functional as F
from common import (V, BOS, SEQ_LEN, id2tok, tok2id, train_model, print_results)


def jacobi_decode(model, max_iters=20, num_samples=1000, use_sampling=False):
    """
    Jacobi decoding: iteratively refine all positions in parallel.

    1. Start with [BOS, random, random, random]
    2. Feed through causal model → get predictions for positions 1, 2, 3
    3. Replace positions 1, 2, 3 with predictions
    4. Repeat until convergence or max_iters
    """
    model.eval()
    results = []
    total_iters = 0
    convergence_counts = {i: 0 for i in range(1, max_iters + 1)}

    content_tokens = [tok2id["I"], tok2id["am"], tok2id["We"], tok2id["are"]]

    with torch.no_grad():
        for _ in range(num_samples):
            # Initialize: BOS + random tokens for positions 1, 2, 3
            seq = [BOS]
            for _ in range(SEQ_LEN - 1):
                seq.append(content_tokens[torch.randint(len(content_tokens), (1,)).item()])
            seq = torch.tensor([seq])  # (1, 4)

            converged_at = max_iters
            for it in range(1, max_iters + 1):
                logits = model(seq)  # (1, 4, V)
                # Position i predicts token i+1, so logits[:, i] → token at position i+1
                # logits[:, 0] → prediction for position 1
                # logits[:, 1] → prediction for position 2
                # logits[:, 2] → prediction for position 3
                if use_sampling:
                    new_tokens = []
                    for pos in range(SEQ_LEN - 1):
                        probs = F.softmax(logits[0, pos], dim=-1)
                        new_tokens.append(torch.multinomial(probs, 1).item())
                else:
                    new_tokens = logits[0, :SEQ_LEN-1].argmax(dim=-1).tolist()

                new_seq = torch.tensor([[BOS] + new_tokens])

                if torch.equal(new_seq, seq):
                    converged_at = it
                    seq = new_seq
                    break
                seq = new_seq

            total_iters += converged_at
            convergence_counts[converged_at] = convergence_counts.get(converged_at, 0) + 1
            results.append(tuple(seq[0].tolist()))

    print(f"Average iterations to converge: {total_iters/num_samples:.2f}")
    print(f"Convergence distribution:")
    for it, count in sorted(convergence_counts.items()):
        if count > 0:
            print(f"  iter {it}: {count} samples ({100*count/num_samples:.1f}%)")
    print()
    return results


if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("JACOBI DECODING")
    print("=" * 60)
    print("Uses the same causal autoregressive model.")
    print("Initialize with random tokens, iteratively refine.\n")

    print("Training causal model...")
    model = train_model(causal=True)
    print()

    print("── Jacobi (argmax) ──")
    results = jacobi_decode(model, num_samples=1000, use_sampling=False)
    print_results(results, 1000)

    print("── Jacobi (sampling) ──")
    results = jacobi_decode(model, num_samples=1000, use_sampling=True)
    print_results(results, 1000)

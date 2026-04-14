"""
Mask-Predict: Iterative masked refinement with a bidirectional model.

Paper: "Mask-Predict: Parallel Decoding of Conditional Masked Language Models"
       (Ghazvininejad et al., 2019)

Core idea:
1. Train a bidirectional (non-causal) transformer with masked language modeling.
2. At inference: start with all positions masked, predict all in parallel.
3. Keep the most confident predictions, re-mask the rest, repeat.

This requires a SEPARATE model trained with bidirectional attention and
a <MASK> token.
"""

import torch
import torch.nn.functional as F
from common import (V, BOS, EOS, MASK, SEQ_LEN, id2tok, tok2id,
                    TinyTransformer, print_results)


def train_mask_predict_model(num_steps=3000):
    """
    Train a bidirectional transformer with masked language modeling.
    For each training sequence, randomly mask 1-3 of the content positions
    (positions 1, 2 — not BOS or EOS) and predict the masked tokens.
    """
    model = TinyTransformer(V, causal=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    seq1 = torch.tensor([BOS, tok2id["I"], tok2id["am"], EOS])
    seq2 = torch.tensor([BOS, tok2id["We"], tok2id["are"], EOS])
    seqs = torch.stack([seq1, seq2])  # (2, 4)

    for step in range(num_steps):
        # Random masking of content positions (1 and 2, not BOS=0 or EOS=3)
        masked = seqs.clone()
        mask_positions = []

        for b in range(2):
            # Mask 1 or 2 content positions randomly
            n_mask = torch.randint(1, 3, (1,)).item()  # 1 or 2
            positions = torch.randperm(2)[:n_mask] + 1  # positions 1 and/or 2
            masked[b, positions] = MASK
            mask_positions.append(positions)

        logits = model(masked)  # (2, 4, V)

        # Loss only on masked positions
        loss = 0
        count = 0
        for b in range(2):
            for pos in mask_positions[b]:
                loss += F.cross_entropy(logits[b, pos], seqs[b, pos])
                count += 1
        loss = loss / count

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 500 == 0:
            print(f"  Step {step+1:4d}  loss={loss.item():.4f}")

    return model


def mask_predict_inference(model, num_iterations=3, num_samples=1000):
    """
    Mask-Predict inference:
    1. Start: [BOS, MASK, MASK, EOS]
    2. Predict all masked positions simultaneously
    3. Keep most confident prediction, re-mask the rest
    4. Repeat until all unmasked
    """
    model.eval()
    results = []

    with torch.no_grad():
        for _ in range(num_samples):
            # Start fully masked (except BOS and EOS)
            seq = torch.tensor([[BOS, MASK, MASK, EOS]])
            masked_positions = [1, 2]  # positions that are currently masked

            for it in range(num_iterations):
                if not masked_positions:
                    break

                logits = model(seq)  # (1, 4, V)

                # Get confidence (max probability) at each masked position
                confidences = {}
                predictions = {}
                for pos in masked_positions:
                    probs = F.softmax(logits[0, pos], dim=-1)
                    max_prob, pred = probs.max(dim=-1)
                    confidences[pos] = max_prob.item()
                    predictions[pos] = pred.item()

                # Unmask the most confident position(s)
                # Schedule: unmask ceil(remaining / (num_iterations - it)) positions
                remaining_iters = num_iterations - it
                n_unmask = max(1, len(masked_positions) // remaining_iters)
                if it == num_iterations - 1:
                    n_unmask = len(masked_positions)  # unmask all on last iter

                sorted_positions = sorted(masked_positions,
                                          key=lambda p: confidences[p],
                                          reverse=True)
                to_unmask = sorted_positions[:n_unmask]

                for pos in to_unmask:
                    seq[0, pos] = predictions[pos]
                    masked_positions.remove(pos)

            results.append(tuple(seq[0].tolist()))

    return results


if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("MASK-PREDICT")
    print("=" * 60)
    print("Bidirectional model trained with masked LM objective.")
    print("Inference: [BOS, MASK, MASK, EOS] → iteratively unmask.\n")

    print("Training bidirectional model...")
    model = train_mask_predict_model(num_steps=3000)
    print()

    for n_iter in [1, 2, 3]:
        print(f"── Mask-Predict ({n_iter} iteration{'s' if n_iter > 1 else ''}) ──")
        results = mask_predict_inference(model, num_iterations=n_iter, num_samples=1000)
        print_results(results, 1000)

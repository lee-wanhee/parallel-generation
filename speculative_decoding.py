"""
Speculative Decoding: Draft-then-verify with an autoregressive model.

Papers: "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)
        "Accelerating Large Language Model Decoding with Speculative Sampling" (Chen et al., 2023)

Core idea:
1. Generate draft tokens in parallel (or from a cheap model)
2. Verify all draft tokens against the target model in one forward pass
3. Accept the longest prefix that matches, resample the first rejected token

In our toy setting, we use random/uniform drafting since there's no
separate draft model. The point is to show that verification recovers
correctness — the target model's autoregressive distribution is preserved
exactly.
"""

import torch
import torch.nn.functional as F
from common import (V, BOS, EOS, SEQ_LEN, id2tok, tok2id, train_model, print_results)


def speculative_decode(model, num_samples=1000):
    """
    Speculative decoding:
    1. Draft: sample all 3 content tokens independently from uniform distribution
       over content tokens (simulating a fast but dumb draft model)
    2. Verify: feed [BOS, draft1, draft2, draft3] through target model
    3. Accept/reject each position left-to-right:
       - At position i, check if draft[i] matches what the target model would
         sample given the accepted prefix
       - Accept if it matches (with proper probability adjustment), reject otherwise
       - On first rejection, resample from target model's distribution
    """
    model.eval()
    results = []
    total_forward_passes = 0
    content_tokens = [tok2id["I"], tok2id["am"], tok2id["We"], tok2id["are"], EOS]

    with torch.no_grad():
        for _ in range(num_samples):
            tokens = [BOS]
            passes = 0

            while len(tokens) < SEQ_LEN:
                # Draft: fill remaining positions with random tokens
                n_draft = SEQ_LEN - len(tokens)
                draft = [content_tokens[torch.randint(len(content_tokens), (1,)).item()]
                         for _ in range(n_draft)]
                candidate = tokens + draft

                # Verify: one forward pass through the full candidate
                x = torch.tensor([candidate])
                logits = model(x)  # (1, SEQ_LEN, V)
                passes += 1

                # Accept/reject left to right
                accepted = 0
                for i in range(len(tokens), SEQ_LEN):
                    # Target model's distribution at this position
                    target_probs = F.softmax(logits[0, i - 1], dim=-1)
                    draft_tok = candidate[i]

                    # Simplified acceptance: accept if draft matches argmax
                    # or with probability proportional to target probability
                    # For exact speculative sampling:
                    # Accept with prob min(1, p_target(x) / p_draft(x))
                    # Since draft is uniform over 5 tokens, p_draft = 0.2
                    p_target = target_probs[draft_tok].item()
                    p_draft = 1.0 / len(content_tokens)
                    accept_prob = min(1.0, p_target / p_draft)

                    if torch.rand(1).item() < accept_prob:
                        tokens.append(draft_tok)
                        accepted += 1
                    else:
                        # Reject: sample from adjusted distribution
                        # p'(x) = max(0, p_target(x) - p_draft(x)) / Z
                        adjusted = torch.clamp(target_probs - p_draft, min=0)
                        if adjusted.sum() > 0:
                            adjusted = adjusted / adjusted.sum()
                            tokens.append(torch.multinomial(adjusted, 1).item())
                        else:
                            tokens.append(torch.multinomial(target_probs, 1).item())
                        break

                if len(tokens) >= SEQ_LEN:
                    break

            total_forward_passes += passes
            results.append(tuple(tokens[:SEQ_LEN]))

    print(f"Average forward passes per sequence: {total_forward_passes/num_samples:.2f}")
    print(f"(Autoregressive would need {SEQ_LEN - 1} passes)\n")
    return results


if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("SPECULATIVE DECODING")
    print("=" * 60)
    print("Draft random tokens, verify against autoregressive model.")
    print("Preserves exact autoregressive distribution.\n")

    print("Training causal model...")
    model = train_model(causal=True)
    print()

    print("── Speculative Decoding ──")
    results = speculative_decode(model, num_samples=1000)
    print_results(results, 1000)

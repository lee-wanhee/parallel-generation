# Parallel Generation: Methods and Results

## Setup

We train tiny transformers (2 layers, 64-dim, 4 heads) on exactly two sequences with equal probability:

| Sequence | Tokens |
|----------|--------|
| 1 | `<BOS> I am <EOS>` |
| 2 | `<BOS> We are <EOS>` |

This is a minimal testbed where the conditional dependency is absolute: knowing token 1 completely determines token 2. Any method that breaks this dependency will produce invalid sequences like "I are" or "We am".

---

## Baseline Results

### Autoregressive Inference
Standard left-to-right generation. Each token is sampled conditioned on all previous tokens.

```
<BOS> I am <EOS>      ~50%
<BOS> We are <EOS>    ~50%
Valid: 100%
```

### Naive Parallel Inference
All positions predicted from `<BOS>` only, sampled independently.

```
<BOS> I I <EOS>       19.2%
<BOS> We I <EOS>      18.5%
<BOS> I We <EOS>      16.4%
...
Valid: <1%
```

Position 2 never sees what was sampled at position 1, so it predicts the marginal distribution (a mixture of "am" and "are") rather than the correct conditional. Result: mostly nonsense.

---

## Method 1: Jacobi Decoding

**Paper lineage**: Lookahead Decoding (ICML 2024), Consistency LLMs (2024)

**Core idea**: Treat autoregressive generation as a fixed-point problem. Initialize all positions with random tokens, feed through the causal model, replace each position with the model's prediction, repeat until convergence.

**Key property**: Uses the SAME causal model — no retraining needed.

### Results (argmax)

```
<BOS> We are <EOS>    100%
Valid: 100%
Average iterations: 2.26
Convergence: 74% at iter 2, 26% at iter 3
```

**Problem: mode collapse.** Argmax Jacobi always converges to the single highest-probability fixed point. Since the model is deterministic once you take argmax, every run converges to the same sequence. With two equally likely sequences, a slight numerical asymmetry makes it always pick "We are".

### Results (sampling)

```
<BOS> We are <EOS>    53.9%
<BOS> I am <EOS>      45.2%
<BOS> We am <EOS>      0.6%
<BOS> I are <EOS>      0.3%
Valid: 99.1%
Average iterations: 5.91
```

Sampling-based Jacobi mostly recovers correct sequences (99.1% valid) but needs more iterations to converge and occasionally produces invalid outputs when sampling creates an unstable cycle.

### Analysis

Jacobi decoding is elegant: it reuses the autoregressive model and converges quickly. However:
- **Argmax variant**: 100% valid but mode-collapses to one sequence
- **Sampling variant**: Diverse but ~1% invalid, and convergence is slow (6 iterations avg vs 2 for argmax)
- **Parallelism**: Each iteration is a single forward pass (all positions computed together), but multiple iterations needed

---

## Method 2: Mask-Predict

**Paper**: "Mask-Predict: Parallel Decoding of Conditional Masked Language Models" (Ghazvininejad et al., 2019)

**Core idea**: Train a bidirectional transformer with masked language modeling. At inference, start with all content positions masked `[BOS, MASK, MASK, EOS]`, predict all masked positions in parallel, keep the most confident predictions, re-mask the rest, and repeat.

**Key property**: Requires a SEPARATE bidirectional model trained with masking.

### Results

```
1 iteration:  <BOS> I am <EOS>   100%    Valid: 100%
2 iterations: <BOS> I am <EOS>   100%    Valid: 100%
3 iterations: <BOS> I am <EOS>   100%    Valid: 100%
```

**Problem: mode collapse (same as Jacobi argmax).** The bidirectional model, when given `[BOS, MASK, MASK, EOS]`, always predicts the same sequence because it uses argmax. The model learns that both "I am" and "We are" are valid, but when both positions are masked simultaneously, it commits to one mode.

Even in a single iteration (fully parallel), the bidirectional model produces 100% valid sequences — it sees both BOS and EOS, allowing it to learn the correlation between positions 1 and 2. However, it only generates one of the two valid sequences.

### Analysis

Mask-Predict solves the validity problem completely — the bidirectional attention lets position 2 attend to position 1 even when both are initially masked, because they are refined together. But:
- **Mode collapse**: argmax-based unmasking always picks the same mode
- **Requires retraining**: Cannot reuse the causal autoregressive model
- **Single iteration works**: For this simple problem, the bidirectional model can resolve dependencies in one pass

---

## Method 3: Speculative Decoding

**Papers**: "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023), "Accelerating LLM Decoding with Speculative Sampling" (Chen et al., 2023)

**Core idea**: Generate draft tokens cheaply (here: random), then verify all of them against the target autoregressive model in a single forward pass. Accept the longest prefix where draft and target agree (with proper probability adjustment). Resample the first rejected position from the adjusted distribution.

**Key property**: Preserves the EXACT autoregressive distribution. Uses the same causal model.

### Results

```
<BOS> I am <EOS>      51.2%
<BOS> We are <EOS>    48.8%
Valid: 100%
Average forward passes: 2.39 (vs 3 for autoregressive)
```

**This is the only method that achieves both 100% validity AND correct diversity** (50/50 split). The verification step guarantees that the output distribution exactly matches autoregressive sampling.

### Analysis

Speculative decoding is the gold standard for preserving quality:
- **100% valid**: Verification catches all invalid drafts
- **Correct distribution**: 50/50 split matches autoregressive exactly
- **Speed**: 2.39 forward passes vs 3 for autoregressive (1.26x speedup)
- **No retraining**: Uses the same causal model
- **Limitation**: Speedup depends on draft quality. Random drafting gives modest gains; a good draft model would do better.

---

## Summary

| Method | Valid % | Diversity | Retraining? | Avg Forward Passes |
|--------|---------|-----------|-------------|-------------------|
| Autoregressive | 100% | 50/50 | — | 3 |
| Naive Parallel | <1% | N/A | No | 1 |
| Jacobi (argmax) | 100% | Mode collapse | No | 2.26 |
| Jacobi (sampling) | 99.1% | ~50/50 | No | 5.91 |
| Mask-Predict | 100% | Mode collapse | Yes | 1 |
| Speculative Decoding | 100% | 50/50 | No | 2.39 |

### Key Takeaways

1. **Naive parallel generation fundamentally breaks conditional dependencies.** Even in a 2-sequence toy problem, it produces <1% valid outputs.

2. **Jacobi decoding** is the simplest fix (no retraining), and the sampling variant achieves 99.1% validity with correct diversity. But convergence is slow and not guaranteed.

3. **Mask-Predict** achieves 100% validity in a single parallel pass thanks to bidirectional attention. But it requires retraining and suffers from mode collapse with argmax decoding.

4. **Speculative decoding** is the only method that achieves both perfect validity and correct diversity without retraining. It is mathematically guaranteed to preserve the autoregressive distribution.

5. **The fundamental tradeoff**: methods that are fully parallel (Mask-Predict, naive) tend toward mode collapse or invalid outputs. Methods that iterate (Jacobi, speculative) recover correctness at the cost of multiple passes. True single-pass parallel generation with correct diversity remains an open problem.

---

## References

- Ghazvininejad et al. "Mask-Predict: Parallel Decoding of Conditional Masked Language Models" (EMNLP 2019)
- Leviathan et al. "Fast Inference from Transformers via Speculative Decoding" (ICML 2023)
- Chen et al. "Accelerating Large Language Model Decoding with Speculative Sampling" (2023)
- Santilli et al. "Accelerating Transformer Inference for Translation via Parallel Decoding" (ACL 2023) — Jacobi decoding
- Fu et al. "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding" (ICML 2024)
- Kou et al. "CLLMs: Consistency Large Language Models" (2024)

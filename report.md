# Parallel Generation: How Fast Can We Resolve Uncertainty?

## Research Question

Given sequences with structured dependencies, what is the minimum number of steps needed to generate a valid sequence? Can parallel generation methods match the theoretical lower bound — the **sequential depth** of the dependency structure?

## Setup

### Structured Uncertainty Datasets

We design sequences as dependency trees. Each sequence has a **sequential depth N**: the minimum number of sequential decisions needed to fully determine all tokens. Tokens below a resolved branch can be generated in parallel.

| Dataset | Depth | # Seqs | Length | Structure |
|---------|-------|--------|--------|-----------|
| independent | 0 | 8 | 5 | All tokens independent — fully parallelizable |
| depth1 | 1 | 2 | 4 | 1 branch → 1 determined token |
| depth1_wide | 1 | 2 | 7 | 1 branch → 4 determined tokens (high parallelism) |
| depth2 | 2 | 4 | 5 | 2 sequential branches → 1 determined token |
| depth3 | 3 | 8 | 6 | 3 sequential branches (binary tree) |
| mixed | 1 or 2 | 6 | 5 | Mix of depth-1 and depth-2 sequences |

**Example — depth1_wide:** After resolving a single branching token (A1 vs A2), four remaining tokens are completely determined and could theoretically be generated in one parallel step. So the theoretical minimum is **2 steps** (1 to resolve the branch + 1 to generate all leaves in parallel), not 5.

### Methods

| Method | Type | Retraining? | Distribution Preserved? |
|--------|------|-------------|------------------------|
| Autoregressive (AR) | Sequential | — | Yes |
| Jacobi (argmax) | Iterative refinement | No | No (mode collapse) |
| Jacobi (sampling) | Iterative refinement | No | Approximately |
| Mask-Predict (MP) | Iterative unmasking | Yes (bidir) | No (mode collapse) |
| Mask-Predict Adaptive (MP-Ad) | Adaptive unmasking | Yes (bidir) | No (mode collapse) |
| Speculative Decoding | Draft & verify | No | Yes (exact) |
| Diffusion (adaptive) | Iterative denoising | Yes (diffusion) | No (mode collapse) |

---

## Results

### Steps to Resolve (lower is better)

| Dataset | Depth | AR | Jac-A | Jac-S | MP | MP-Ad | Spec | Diff |
|---------|-------|----|-------|-------|-----|-------|------|------|
| independent | 0 | 3.00 | 2.30 | 8.29 | 3.00 | 3.00 | 3.01 | 3.00 |
| depth1 | 1 | 2.00 | 2.42 | 5.75! | 2.00 | 2.00 | 2.29 | 2.00 |
| **depth1_wide** | **1** | **5.00** | **4.70** | **17.48!** | **5.00** | **2.00** | **5.39** | **2.00** |
| depth2 | 2 | 3.00 | 3.79 | 16.48! | 3.00 | 3.00 | 3.52 | 3.00 |
| depth3 | 3 | 4.00 | 3.77 | 19.77! | 4.00 | 4.00 | 4.67 | 4.00 |
| mixed | 1-2 | 3.00! | 3.92 | 19.14! | 3.00 | 3.00 | 3.60 | 2.00 |

`!` = <100% validity

### Validity

| Dataset | AR | Jac-A | Jac-S | MP | MP-Ad | Spec | Diff |
|---------|-----|-------|-------|-----|-------|------|------|
| independent | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| depth1 | 100% | 100% | 99.5% | 100% | 100% | 100% | 100% |
| depth1_wide | 100% | 100% | 30.0% | 100% | 100% | 100% | 100% |
| depth2 | 100% | 100% | 42.0% | 100% | 100% | 100% | 100% |
| depth3 | 100% | 100% | 4.5% | 100% | 100% | 100% | 100% |
| mixed | 99.5% | 100% | 15.5% | 100% | 100% | 100% | 100% |

### Diversity (bits of entropy; higher = more diverse)

| Dataset | Max | AR | Jac-A | Jac-S | MP | MP-Ad | Spec | Diff |
|---------|-----|----|-------|-------|-----|-------|------|------|
| independent | 3.00 | 2.98 | 0 | 2.94 | 0 | 0 | 2.98 | 0 |
| depth1 | 1.00 | 0.99 | 0 | 1.04 | 0 | 0 | 1.00 | 0 |
| depth1_wide | 1.00 | 1.00 | 0 | 4.43 | 0 | 0 | 1.00 | 0 |
| depth2 | 2.00 | 1.97 | 0 | 4.51 | 0 | 0 | 2.00 | 0 |
| depth3 | 3.00 | 2.98 | 0 | 7.26 | 0 | 0 | 2.99 | 0 |
| mixed | 2.58 | 2.59 | 0 | 6.40 | 0 | 0 | 2.56 | 0 |

---

## Analysis

### Key Finding 1: Adaptive methods exploit parallel structure

The **depth1_wide** dataset is the critical test. It has 5 content tokens but only 1 branching decision — after resolving the first token, 4 tokens are determined and can be generated in parallel.

- **AR takes 5 steps** (one per token, left-to-right — blind to the parallel structure)
- **Mask-Predict Adaptive and Diffusion take 2 steps** — they resolve the branch, then generate all 4 leaves in one parallel step
- This matches the **theoretical minimum** of depth + 1 = 2

This demonstrates that **bidirectional methods (MP-Adaptive, Diffusion) can exploit parallel structure** that autoregressive methods cannot.

### Key Finding 2: Most methods match depth + 1 on chain structures

For pure chain dependencies (depth1, depth2, depth3), most methods achieve steps ≈ depth + 1:
- AR: exactly `num_content_tokens` (no parallelism exploited)
- MP, MP-Adaptive, Diffusion: exactly `depth + 1` (matches theoretical minimum on chains)
- Jacobi-argmax: close to `depth + 1` but slightly higher
- Speculative: slightly above AR (overhead from rejected drafts)

### Key Finding 3: The validity-diversity-speed trilemma

No single method achieves all three of: (1) 100% validity, (2) correct diversity, (3) minimum steps.

| | Valid | Diverse | Fast |
|--|-------|---------|------|
| AR | Yes | Yes | No (always sequential) |
| Jacobi-argmax | Yes | **No** (mode collapse) | Yes |
| Jacobi-sampling | **No** (degrades with depth) | Yes | **No** (many iterations) |
| MP / MP-Ad / Diff | Yes | **No** (mode collapse) | **Yes** (matches depth+1) |
| Speculative | Yes | Yes | No (overhead from random drafts) |

### Key Finding 4: Mode collapse is the central problem

Jacobi-argmax, Mask-Predict, and Diffusion all achieve 0 bits of entropy — they always produce the same sequence. They use argmax decoding, which selects a single fixed point.

This is fundamentally different from the AR model, which naturally samples from the correct distribution. **The speed advantage of parallel methods comes at the cost of diversity.**

### Key Finding 5: Jacobi-sampling breaks on deeper structures

Jacobi with sampling maintains diversity but validity drops sharply with depth:
- depth1: 99.5% valid
- depth2: 42.0% valid
- depth3: 4.5% valid

The sampling creates unstable oscillations that prevent convergence to valid fixed points.

---

## Towards Optimal Uncertainty Resolution

### The entropy-reduction hypothesis

The ideal method should **maximize entropy reduction per step**:

1. **Identify the highest-entropy token** — the one whose resolution unlocks the most downstream tokens
2. **Resolve it** — sample from its distribution
3. **Propagate** — re-evaluate all remaining tokens given the new information
4. **Generate in parallel** all tokens that are now determined (entropy ≈ 0)

This is essentially what Mask-Predict Adaptive and Diffusion do when they achieve the depth+1 lower bound. But they lack the sampling step (2) — they use argmax instead of sampling, causing mode collapse.

### What's missing: Sampling + Parallelism

The open problem is combining:
- **Sampling** from the correct joint distribution (not just argmax)
- **Parallel generation** of conditionally-determined tokens
- **Adaptive scheduling** based on current entropy of each position

A method that could do all three would:
1. Sample the highest-entropy position (e.g., position 1 in depth1_wide)
2. In the same or next step, generate all positions that become deterministic (positions 2-5 in depth1_wide) in parallel
3. Achieve depth+1 steps with correct diversity

### Possible approaches to investigate

1. **Sampling-aware Mask-Predict**: Instead of argmax unmasking, sample the most uncertain position, then unmask all high-confidence positions in the next step.

2. **Entropy-guided iterative refinement**: At each step, measure per-position entropy. Sample positions with entropy above a threshold. Commit positions with entropy near zero.

3. **Speculative decoding with adaptive draft**: Instead of random drafts, use a bidirectional model to draft all positions, then verify with the AR model. The draft would be high-quality (exploiting parallel structure) and verification would preserve the correct distribution.

4. **Diffusion with stochastic denoising**: Replace argmax in the diffusion reverse process with proper sampling, adjusted to maintain the correct marginals at each step.

---

## References

- Ghazvininejad et al. "Mask-Predict: Parallel Decoding of Conditional Masked Language Models" (EMNLP 2019)
- Leviathan et al. "Fast Inference from Transformers via Speculative Decoding" (ICML 2023)
- Chen et al. "Accelerating Large Language Model Decoding with Speculative Sampling" (2023)
- Santilli et al. "Accelerating Transformer Inference via Parallel Decoding" (ACL 2023)
- Fu et al. "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding" (ICML 2024)
- Kou et al. "CLLMs: Consistency Large Language Models" (2024)
- Austin et al. "Structured Denoising Diffusion Models in Discrete State-Spaces" (2021)

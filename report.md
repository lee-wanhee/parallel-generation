# Autoregressive vs Parallel Generation: Toy Experiment

## Setup

We train a small autoregressive transformer (2 layers, 64-dim, 4 heads) on exactly two sequences with equal probability:

| Sequence | Tokens |
|----------|--------|
| 1 | `<BOS> I am <EOS>` |
| 2 | `<BOS> We are <EOS>` |

The model is trained with standard next-token prediction loss for 2000 steps, converging to a loss of 0.2311 (close to the theoretical minimum of `ln(2) ≈ 0.2310` for the first-token prediction, since all other tokens are deterministic given context).

## Autoregressive Inference

Standard left-to-right generation: sample one token, feed it back, sample the next.

```
<BOS> We are <EOS>    509  (50.9%)
<BOS> I am <EOS>      491  (49.1%)
```

The model produces only valid sequences, with a near-perfect 50/50 split. This is expected: at position 1, the model samples "I" or "We" with equal probability. At position 2, it conditions on the sampled token — if it sees "I", it deterministically outputs "am"; if "We", it outputs "are".

## Parallel Inference

All positions are predicted from `<BOS>` simultaneously, then sampled independently. We feed `[BOS, BOS, BOS, BOS]` so that each position (via causal masking) only attends to BOS tokens, then sample positions 1, 2, and 3 independently.

### Per-position distributions

| Position | Top predictions |
|----------|----------------|
| 1 (should be I/We) | I=0.500, We=0.500 |
| 2 (should be am/are) | I=0.556, We=0.432, am=0.010 |
| 3 (should be EOS) | EOS=0.730, I=0.147, We=0.122 |

### Generated sequences

```
<BOS> I I <EOS>       192  (19.2%)
<BOS> We I <EOS>      185  (18.5%)
<BOS> I We <EOS>      164  (16.4%)
<BOS> We We <EOS>     156  (15.6%)
<BOS> We I I           51  (5.1%)
<BOS> I I I            47  (4.7%)
...
<BOS> I am <EOS>        4  (0.4%)
<BOS> We are <EOS>      2  (0.2%)
```

Valid sequences appear less than 1% of the time. The vast majority of outputs are nonsensical.

## Why Parallel Fails

The core issue is that parallel generation breaks **conditional dependencies** between tokens.

In the training data, the joint distribution factorizes as:

```
P("I am")  = P("I" | BOS) × P("am" | BOS, "I")  = 0.5 × 1.0 = 0.5
P("We are") = P("We" | BOS) × P("are" | BOS, "We") = 0.5 × 1.0 = 0.5
```

The conditional `P(token₂ | token₁)` is deterministic — "I" always implies "am", "We" always implies "are". Autoregressive generation preserves this dependency because each token is sampled conditioned on all previous tokens.

Parallel generation instead computes:

```
P_parallel(token₁, token₂) = P(token₁ | BOS) × P(token₂ | BOS)
```

Position 2 never sees what was sampled at position 1. It only sees BOS, so it must predict the **marginal** distribution over position 2 — a mixture of "am" (given "I") and "are" (given "We"). In practice, the model collapses to predicting high-frequency tokens ("I", "We") rather than the correct conditionals, because the BOS-only context gives it no signal about which branch was taken.

This is a fundamental limitation: parallel generation assumes token positions are conditionally independent given the prompt, but natural language has strong inter-token dependencies. Even in this minimal two-sequence example, the dependencies are absolute — knowing token 1 completely determines token 2.

## Implications

This experiment provides a minimal proof that naive parallel generation (predicting all tokens from the same context simultaneously) cannot preserve the sequential dependencies that autoregressive models learn. Any approach to parallel generation must address this dependency problem, for example through:

- **Iterative refinement**: Generate all tokens in parallel, then iteratively update them conditioned on each other (e.g., BERT-style masked prediction, diffusion)
- **Latent planning**: Predict a latent variable that captures the global choice ("I am" vs "We are") before generating tokens
- **Consistency training**: Train the model to produce coherent parallel outputs directly

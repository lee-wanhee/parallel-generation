"""
Structured Uncertainty Datasets

Each dataset has sequences with a specific dependency structure.
The key parameter is the "sequential depth" N — the minimum number
of sequential decisions needed to fully resolve the sequence.

Dataset design principle:
- Some tokens are "branching decisions" that must be resolved sequentially
- Other tokens are deterministic once the branching decisions are known
- The goal: find methods that resolve uncertainty in exactly N steps

All sequences have format: [BOS, tok1, tok2, ..., tokK, EOS]
"""

import torch
from itertools import product
import random


def make_vocab(token_names):
    """Create vocab from token names, always including BOS, EOS, MASK."""
    special = ["<BOS>", "<EOS>", "<MASK>"]
    all_tokens = special + token_names
    tok2id = {t: i for i, t in enumerate(all_tokens)}
    id2tok = {i: t for t, i in tok2id.items()}
    return tok2id, id2tok, len(all_tokens)


# ═══════════════════════════════════════════════════════════════════
# Dataset 1: Depth-1 (our original "I am / We are")
#   One branching decision at position 1 determines everything else.
#   Sequential depth N=1.
# ═══════════════════════════════════════════════════════════════════

def dataset_depth1():
    """
    Two sequences:
      [BOS] A1 B1 [EOS]    (50%)
      [BOS] A2 B2 [EOS]    (50%)

    Dependency: position 1 → position 2 (deterministic)
    Sequential depth: 1 (resolve position 1, then position 2 is determined)
    """
    token_names = ["A1", "B1", "A2", "B2"]
    tok2id, id2tok, V = make_vocab(token_names)

    sequences = [
        [tok2id["<BOS>"], tok2id["A1"], tok2id["B1"], tok2id["<EOS>"]],
        [tok2id["<BOS>"], tok2id["A2"], tok2id["B2"], tok2id["<EOS>"]],
    ]

    return {
        "name": "depth1",
        "description": "1 branching decision → 1 determined token",
        "sequential_depth": 1,
        "tok2id": tok2id, "id2tok": id2tok, "V": V,
        "sequences": sequences,
        "seq_len": 4,
        "content_positions": [1, 2],  # positions that need to be generated
    }


# ═══════════════════════════════════════════════════════════════════
# Dataset 2: Depth-1 with parallel leaves
#   One branching decision determines MULTIPLE downstream tokens.
#   Sequential depth N=1, but more parallelism potential.
# ═══════════════════════════════════════════════════════════════════

def dataset_depth1_wide():
    """
    Two sequences, 5 content tokens each:
      [BOS] A1 B1 C1 D1 E1 [EOS]    (50%)
      [BOS] A2 B2 C2 D2 E2 [EOS]    (50%)

    Dependency: position 1 → positions 2,3,4,5 (all deterministic)
    Sequential depth: 1
    Parallelism: Once position 1 is resolved, 4 tokens can be generated in parallel.
    """
    token_names = [f"{letter}{i}" for i in [1, 2] for letter in "ABCDE"]
    tok2id, id2tok, V = make_vocab(token_names)

    sequences = [
        [tok2id["<BOS>"]] + [tok2id[f"{l}1"] for l in "ABCDE"] + [tok2id["<EOS>"]],
        [tok2id["<BOS>"]] + [tok2id[f"{l}2"] for l in "ABCDE"] + [tok2id["<EOS>"]],
    ]

    return {
        "name": "depth1_wide",
        "description": "1 branching decision → 4 determined tokens (high parallelism)",
        "sequential_depth": 1,
        "tok2id": tok2id, "id2tok": id2tok, "V": V,
        "sequences": sequences,
        "seq_len": 7,
        "content_positions": list(range(1, 6)),
    }


# ═══════════════════════════════════════════════════════════════════
# Dataset 3: Depth-2 (chain)
#   Two sequential branching decisions.
#   Position 1 has 2 options. Given position 1, position 2 has 2 options.
#   Position 3 is determined by positions 1 and 2.
#   Sequential depth N=2.
# ═══════════════════════════════════════════════════════════════════

def dataset_depth2():
    """
    Four sequences (2 × 2 branching):
      [BOS] A1 B1 C1 [EOS]    (25%)  — branch A1, then B1
      [BOS] A1 B2 C2 [EOS]    (25%)  — branch A1, then B2
      [BOS] A2 B3 C3 [EOS]    (25%)  — branch A2, then B3
      [BOS] A2 B4 C4 [EOS]    (25%)  — branch A2, then B4

    Dependency: pos1 → pos2 → pos3
    Sequential depth: 2
    """
    token_names = ["A1", "A2", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4"]
    tok2id, id2tok, V = make_vocab(token_names)

    sequences = [
        [tok2id["<BOS>"], tok2id["A1"], tok2id["B1"], tok2id["C1"], tok2id["<EOS>"]],
        [tok2id["<BOS>"], tok2id["A1"], tok2id["B2"], tok2id["C2"], tok2id["<EOS>"]],
        [tok2id["<BOS>"], tok2id["A2"], tok2id["B3"], tok2id["C3"], tok2id["<EOS>"]],
        [tok2id["<BOS>"], tok2id["A2"], tok2id["B4"], tok2id["C4"], tok2id["<EOS>"]],
    ]

    return {
        "name": "depth2",
        "description": "2 sequential branching decisions → 1 determined token",
        "sequential_depth": 2,
        "tok2id": tok2id, "id2tok": id2tok, "V": V,
        "sequences": sequences,
        "seq_len": 5,
        "content_positions": [1, 2, 3],
    }


# ═══════════════════════════════════════════════════════════════════
# Dataset 4: Depth-2 with parallel leaves
#   Two branching decisions, each followed by parallel deterministic tokens.
#   Sequential depth N=2, but with significant parallelism.
# ═══════════════════════════════════════════════════════════════════

def dataset_depth2_wide():
    """
    Four sequences:
      [BOS] A1 B1 C1 D1 E1 F1 [EOS]   — branch A1→B1, then C1,D1,E1,F1 determined
      [BOS] A1 B2 C2 D2 E2 F2 [EOS]   — branch A1→B2
      [BOS] A2 B3 C3 D3 E3 F3 [EOS]   — branch A2→B3
      [BOS] A2 B4 C4 D4 E4 F4 [EOS]   — branch A2→B4

    Dependency: pos1 → pos2 → {pos3, pos4, pos5, pos6} (parallel)
    Sequential depth: 2
    After 2 decisions: 4 tokens can be generated in parallel.
    """
    token_names = []
    for i in range(1, 3):  # A1, A2
        token_names.append(f"A{i}")
    for i in range(1, 5):  # B1-B4
        token_names.append(f"B{i}")
    for letter in "CDEF":
        for i in range(1, 5):
            token_names.append(f"{letter}{i}")

    tok2id, id2tok, V = make_vocab(token_names)

    sequences = []
    for branch_idx, (a, b) in enumerate([(1,1), (1,2), (2,3), (2,4)]):
        seq = [tok2id["<BOS>"], tok2id[f"A{a}"], tok2id[f"B{b}"]]
        seq += [tok2id[f"{letter}{b}"] for letter in "CDEF"]
        seq.append(tok2id["<EOS>"])
        sequences.append(seq)

    return {
        "name": "depth2_wide",
        "description": "2 branching decisions → 4 parallel determined tokens",
        "sequential_depth": 2,
        "tok2id": tok2id, "id2tok": id2tok, "V": V,
        "sequences": sequences,
        "seq_len": 8,
        "content_positions": list(range(1, 7)),
    }


# ═══════════════════════════════════════════════════════════════════
# Dataset 5: Depth-3 (deep chain)
#   Three sequential branching decisions (binary tree of depth 3).
#   8 possible sequences.
#   Sequential depth N=3.
# ═══════════════════════════════════════════════════════════════════

def dataset_depth3():
    """
    8 sequences from a binary tree of depth 3:
      pos1 ∈ {X1, X2}
      pos2 ∈ {Y1, Y2, Y3, Y4}  (2 options per pos1 value)
      pos3 ∈ {Z1..Z8}          (2 options per pos2 value)
      pos4 = determined by pos3

    Sequential depth: 3
    """
    token_names = []
    for i in range(1, 3): token_names.append(f"X{i}")
    for i in range(1, 5): token_names.append(f"Y{i}")
    for i in range(1, 9): token_names.append(f"Z{i}")
    for i in range(1, 9): token_names.append(f"W{i}")

    tok2id, id2tok, V = make_vocab(token_names)

    # Binary tree: X1→{Y1,Y2}, X2→{Y3,Y4}, Y1→{Z1,Z2}, etc.
    sequences = []
    z_idx = 1
    for x in [1, 2]:
        for y_offset in [0, 1]:
            y = (x - 1) * 2 + y_offset + 1
            for z_offset in [0, 1]:
                z = z_idx
                seq = [tok2id["<BOS>"], tok2id[f"X{x}"], tok2id[f"Y{y}"],
                       tok2id[f"Z{z}"], tok2id[f"W{z}"], tok2id["<EOS>"]]
                sequences.append(seq)
                z_idx += 1

    return {
        "name": "depth3",
        "description": "3 sequential branching decisions (binary tree depth 3)",
        "sequential_depth": 3,
        "tok2id": tok2id, "id2tok": id2tok, "V": V,
        "sequences": sequences,
        "seq_len": 6,
        "content_positions": [1, 2, 3, 4],
    }


# ═══════════════════════════════════════════════════════════════════
# Dataset 6: Fully parallel (depth-0)
#   All tokens are independent — no dependencies.
#   Sequential depth N=0 (should be solvable in 1 parallel step).
# ═══════════════════════════════════════════════════════════════════

def dataset_independent():
    """
    Each position independently and uniformly chooses from 2 options.
    All 2^3 = 8 combinations are valid.

    [BOS] {A1|A2} {B1|B2} {C1|C2} [EOS]

    No dependencies between positions.
    Sequential depth: 0 (all positions can be resolved in parallel).
    """
    token_names = ["A1", "A2", "B1", "B2", "C1", "C2"]
    tok2id, id2tok, V = make_vocab(token_names)

    sequences = []
    for a in [1, 2]:
        for b in [1, 2]:
            for c in [1, 2]:
                seq = [tok2id["<BOS>"], tok2id[f"A{a}"], tok2id[f"B{b}"],
                       tok2id[f"C{c}"], tok2id["<EOS>"]]
                sequences.append(seq)

    return {
        "name": "independent",
        "description": "All tokens independent (depth 0, fully parallel)",
        "sequential_depth": 0,
        "tok2id": tok2id, "id2tok": id2tok, "V": V,
        "sequences": sequences,
        "seq_len": 5,
        "content_positions": [1, 2, 3],
    }


# ═══════════════════════════════════════════════════════════════════
# Dataset 7: Mixed — different samples have different depths
#   Some sequences need 1 step, some need 2.
#   This tests whether a method can adaptively use fewer steps
#   for simpler samples.
# ═══════════════════════════════════════════════════════════════════

def dataset_mixed():
    """
    Mix of depth-1 and depth-2 sequences:

    Depth-1 patterns (resolve pos1, rest determined):
      [BOS] A1 B1 C1 [EOS]   (25%)
      [BOS] A2 B2 C2 [EOS]   (25%)

    Depth-2 patterns (resolve pos1 AND pos2, then pos3 determined):
      [BOS] A3 B3 C3 [EOS]   (12.5%)
      [BOS] A3 B4 C4 [EOS]   (12.5%)
      [BOS] A4 B5 C5 [EOS]   (12.5%)
      [BOS] A4 B6 C6 [EOS]   (12.5%)

    A method should be able to detect that A1/A2 sequences only need
    1 step while A3/A4 need 2.
    """
    token_names = [f"A{i}" for i in range(1, 5)]
    token_names += [f"B{i}" for i in range(1, 7)]
    token_names += [f"C{i}" for i in range(1, 7)]
    tok2id, id2tok, V = make_vocab(token_names)

    sequences = [
        # Depth-1: A determines B and C
        [tok2id["<BOS>"], tok2id["A1"], tok2id["B1"], tok2id["C1"], tok2id["<EOS>"]],
        [tok2id["<BOS>"], tok2id["A2"], tok2id["B2"], tok2id["C2"], tok2id["<EOS>"]],
        # Depth-2: A determines B options, B determines C
        [tok2id["<BOS>"], tok2id["A3"], tok2id["B3"], tok2id["C3"], tok2id["<EOS>"]],
        [tok2id["<BOS>"], tok2id["A3"], tok2id["B4"], tok2id["C4"], tok2id["<EOS>"]],
        [tok2id["<BOS>"], tok2id["A4"], tok2id["B5"], tok2id["C5"], tok2id["<EOS>"]],
        [tok2id["<BOS>"], tok2id["A4"], tok2id["B6"], tok2id["C6"], tok2id["<EOS>"]],
    ]

    # Per-sequence depth info for evaluation
    per_seq_depth = [1, 1, 2, 2, 2, 2]

    return {
        "name": "mixed",
        "description": "Mix of depth-1 and depth-2 sequences",
        "sequential_depth": "mixed (1 or 2)",
        "per_seq_depth": per_seq_depth,
        "tok2id": tok2id, "id2tok": id2tok, "V": V,
        "sequences": sequences,
        "seq_len": 5,
        "content_positions": [1, 2, 3],
    }


ALL_DATASETS = [
    dataset_independent,
    dataset_depth1,
    dataset_depth1_wide,
    dataset_depth2,
    dataset_depth2_wide,
    dataset_depth3,
    dataset_mixed,
]


if __name__ == "__main__":
    print("Available datasets:\n")
    for fn in ALL_DATASETS:
        ds = fn()
        print(f"  {ds['name']:20s}  depth={str(ds['sequential_depth']):10s}  "
              f"seqs={len(ds['sequences']):2d}  len={ds['seq_len']}  V={ds['V']}")
        for seq in ds["sequences"][:4]:
            tokens = [ds["id2tok"][t] for t in seq]
            print(f"    {' '.join(tokens)}")
        if len(ds["sequences"]) > 4:
            print(f"    ... ({len(ds['sequences']) - 4} more)")
        print()

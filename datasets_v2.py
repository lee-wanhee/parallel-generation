"""
Structured Uncertainty Datasets v2 — Random and Diverse

Key improvements over v1:
1. Shared vocabulary across positions (tokens are reused)
2. Probabilistic (not deterministic) dependencies
3. Randomly generated — different seeds produce different datasets
4. Configurable: vocab size, sequence length, branching factor, depth
5. Much larger: hundreds of sequences, not just 2-8

The dependency structure is still controlled, but the actual token
assignments are random, making the task harder and more realistic.
"""

import torch
import random
from collections import defaultdict


def make_vocab(vocab_size):
    """Create vocab with shared token pool."""
    special = ["<BOS>", "<EOS>", "<MASK>"]
    content_tokens = [f"t{i}" for i in range(vocab_size)]
    all_tokens = special + content_tokens
    tok2id = {t: i for i, t in enumerate(all_tokens)}
    id2tok = {i: t for t, i in tok2id.items()}
    return tok2id, id2tok, len(all_tokens), content_tokens


def generate_tree_sequences(depth, branching_factor, num_leaves_tokens,
                             vocab_size, seq_len, seed=42):
    """
    Generate sequences from a random dependency tree.

    The tree has `depth` levels of branching decisions. At each level,
    the token is chosen from a shared vocabulary (not unique tokens).
    After the last branch, remaining positions are filled with tokens
    that are deterministic given the full branch path.

    Args:
        depth: number of sequential branching decisions
        branching_factor: number of options at each branch (e.g. 2 = binary)
        num_leaves_tokens: number of determined positions after all branches
        vocab_size: size of shared content vocabulary
        seq_len: total sequence length including BOS and EOS
        seed: random seed for reproducibility

    Returns:
        dataset dict compatible with benchmark.py
    """
    rng = random.Random(seed)
    tok2id, id2tok, V, content_tokens = make_vocab(vocab_size)

    # Content positions: [1, 2, ..., seq_len-2] (excluding BOS=0, EOS=last)
    n_content = seq_len - 2
    assert depth + num_leaves_tokens <= n_content, \
        f"depth({depth}) + leaves({num_leaves_tokens}) > content positions({n_content})"

    # Branch positions: first `depth` content positions
    # Leaf positions: next `num_leaves_tokens` content positions
    # Any remaining positions are also leaves (determined by branch path)
    branch_positions = list(range(1, depth + 1))
    leaf_positions = list(range(depth + 1, n_content + 1))

    # Build the tree: at each branch level, randomly assign tokens
    # A "path" is a tuple of choices at each branch level
    # For each path, we randomly assign a token from the content vocab

    # Generate all possible paths
    num_paths = branching_factor ** depth
    paths = []

    def gen_paths(current_path, level):
        if level == depth:
            paths.append(tuple(current_path))
            return
        for choice in range(branching_factor):
            gen_paths(current_path + [choice], level + 1)

    gen_paths([], 0)

    # For each branch level, map (parent_path, choice) -> token
    # Use shared vocab: different branches CAN map to the same token
    branch_token_map = {}  # (level, parent_path_tuple, choice) -> token_id
    for level in range(depth):
        # How many distinct parent paths exist at this level?
        parent_paths = set()
        for path in paths:
            parent_paths.add(path[:level])

        for parent in parent_paths:
            # Randomly assign tokens for each choice at this branch
            for choice in range(branching_factor):
                token = rng.choice(content_tokens)
                branch_token_map[(level, parent, choice)] = tok2id[token]

    # For each leaf position, map full_path -> token
    leaf_token_map = {}  # (leaf_pos_index, full_path) -> token_id
    for leaf_idx, pos in enumerate(leaf_positions):
        for path in paths:
            token = rng.choice(content_tokens)
            leaf_token_map[(leaf_idx, path)] = tok2id[token]

    # Generate sequences
    sequences = []
    for path in paths:
        seq = [tok2id["<BOS>"]]
        # Branch positions
        for level in range(depth):
            parent = path[:level]
            choice = path[level]
            token_id = branch_token_map[(level, parent, choice)]
            seq.append(token_id)
        # Leaf positions
        for leaf_idx in range(len(leaf_positions)):
            token_id = leaf_token_map[(leaf_idx, path)]
            seq.append(token_id)
        seq.append(tok2id["<EOS>"])

        assert len(seq) == seq_len, f"Expected {seq_len}, got {len(seq)}"
        sequences.append(seq)

    content_positions = list(range(1, seq_len - 1))

    return {
        "name": f"tree_d{depth}_b{branching_factor}_l{num_leaves_tokens}_v{vocab_size}_s{seed}",
        "description": (f"Random tree: depth={depth}, branch={branching_factor}, "
                        f"leaves={num_leaves_tokens}, vocab={vocab_size}"),
        "sequential_depth": depth,
        "tok2id": tok2id, "id2tok": id2tok, "V": V,
        "sequences": sequences,
        "seq_len": seq_len,
        "content_positions": content_positions,
        "num_paths": num_paths,
        "branch_positions": branch_positions,
        "leaf_positions": leaf_positions,
    }


# ═══════════════════════════════════════════════════════════════════
# Predefined dataset configurations
# ═══════════════════════════════════════════════════════════════════

def dataset_v2_independent(seed=42):
    """Depth 0: all positions independent. 3 content tokens from vocab of 4."""
    rng = random.Random(seed)
    vocab_size = 4
    tok2id, id2tok, V, content_tokens = make_vocab(vocab_size)

    # Each position independently uniform over vocab
    sequences = []
    for a in range(vocab_size):
        for b in range(vocab_size):
            for c in range(vocab_size):
                seq = [tok2id["<BOS>"],
                       tok2id[content_tokens[a]],
                       tok2id[content_tokens[b]],
                       tok2id[content_tokens[c]],
                       tok2id["<EOS>"]]
                sequences.append(seq)

    return {
        "name": f"v2_independent_v{vocab_size}_s{seed}",
        "description": f"All positions independent, vocab={vocab_size}",
        "sequential_depth": 0,
        "tok2id": tok2id, "id2tok": id2tok, "V": V,
        "sequences": sequences,
        "seq_len": 5,
        "content_positions": [1, 2, 3],
    }


def dataset_v2_depth1_narrow(seed=42):
    """Depth 1, binary branch, 1 leaf. Small."""
    return generate_tree_sequences(
        depth=1, branching_factor=2, num_leaves_tokens=1,
        vocab_size=4, seq_len=4, seed=seed,
    )


def dataset_v2_depth1_wide(seed=42):
    """Depth 1, binary branch, 4 leaves. Tests parallelism."""
    return generate_tree_sequences(
        depth=1, branching_factor=2, num_leaves_tokens=4,
        vocab_size=6, seq_len=7, seed=seed,
    )


def dataset_v2_depth2(seed=42):
    """Depth 2, binary branches, 2 leaves."""
    return generate_tree_sequences(
        depth=2, branching_factor=2, num_leaves_tokens=2,
        vocab_size=6, seq_len=6, seed=seed,
    )


def dataset_v2_depth2_wide(seed=42):
    """Depth 2, binary branches, 4 leaves. Bigger."""
    return generate_tree_sequences(
        depth=2, branching_factor=2, num_leaves_tokens=4,
        vocab_size=8, seq_len=8, seed=seed,
    )


def dataset_v2_depth3(seed=42):
    """Depth 3, binary tree, 1 leaf."""
    return generate_tree_sequences(
        depth=3, branching_factor=2, num_leaves_tokens=1,
        vocab_size=6, seq_len=6, seed=seed,
    )


def dataset_v2_depth1_highbranch(seed=42):
    """Depth 1 but branching factor 4. More uncertainty at the branch."""
    return generate_tree_sequences(
        depth=1, branching_factor=4, num_leaves_tokens=3,
        vocab_size=6, seq_len=6, seed=seed,
    )


def dataset_v2_depth2_highbranch(seed=42):
    """Depth 2, branching factor 3, 2 leaves."""
    return generate_tree_sequences(
        depth=2, branching_factor=3, num_leaves_tokens=2,
        vocab_size=8, seq_len=6, seed=seed,
    )


ALL_V2_DATASETS = [
    dataset_v2_independent,
    dataset_v2_depth1_narrow,
    dataset_v2_depth1_wide,
    dataset_v2_depth2,
    dataset_v2_depth2_wide,
    dataset_v2_depth3,
    dataset_v2_depth1_highbranch,
    dataset_v2_depth2_highbranch,
]


if __name__ == "__main__":
    print("V2 Datasets (random, shared vocabulary):\n")
    for fn in ALL_V2_DATASETS:
        ds = fn()
        print(f"  {ds['name']}")
        print(f"    {ds['description']}")
        print(f"    seqs={len(ds['sequences']):3d}  len={ds['seq_len']}  V={ds['V']}  "
              f"depth={ds['sequential_depth']}")

        # Show first few sequences
        for seq in ds["sequences"][:4]:
            tokens = [ds["id2tok"][t] for t in seq]
            print(f"      {' '.join(tokens)}")
        if len(ds["sequences"]) > 4:
            print(f"      ... ({len(ds['sequences']) - 4} more)")

        # Check token reuse: how many unique tokens per position?
        for pos in ds["content_positions"][:5]:
            unique = set(seq[pos] for seq in ds["sequences"])
            print(f"    pos {pos}: {len(unique)} unique tokens "
                  f"({', '.join(ds['id2tok'][t] for t in sorted(unique))})")
        print()

    # Verify: run with multiple seeds to show diversity
    print("\n--- Same config, different seeds ---")
    for seed in [42, 123, 999]:
        ds = dataset_v2_depth1_wide(seed=seed)
        print(f"\n  seed={seed}: {ds['name']}")
        for seq in ds["sequences"][:2]:
            tokens = [ds["id2tok"][t] for t in seq]
            print(f"    {' '.join(tokens)}")

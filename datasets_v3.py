"""
Structured Uncertainty Datasets v3 — DAG-based Dependencies

Each token position has an explicit dependency set: the set of other
positions that determine its distribution. The dependency structure
forms a DAG (directed acyclic graph).

Key properties:
- The sequential depth = longest path in the DAG = minimum sequential steps
- Tokens with no unresolved dependencies can be generated in parallel
- Shared vocabulary makes it non-trivial to learn
- Random generation ensures robustness

The theoretical minimum steps to generate a valid sequence equals
the number of "levels" when you do a topological sort of the DAG
(i.e., the critical path length + 1 for BOS).

Dependency Structures:
1. Independent: no edges — all tokens can be generated in 1 parallel step
2. Chain: A→B→C→D — purely sequential, depth = num_tokens
3. Tree: A→{B,C,D} — one root, everything else is a leaf
4. Diamond: A→{B,C}→D — convergence point, D depends on two independent tokens
5. Wide diamond: A→{B,C,D,E}→F — many parallel paths converging
6. Two-level: {A,B}→{C,D} — two independent roots, two dependent tokens
7. Custom DAGs: arbitrary structures
"""

import torch
import random
from collections import defaultdict, deque
from itertools import product


def make_vocab(vocab_size):
    special = ["<BOS>", "<EOS>", "<MASK>"]
    content_tokens = [f"t{i}" for i in range(vocab_size)]
    all_tokens = special + content_tokens
    tok2id = {t: i for i, t in enumerate(all_tokens)}
    id2tok = {i: t for t, i in tok2id.items()}
    return tok2id, id2tok, len(all_tokens), content_tokens


def compute_dag_depth(edges, positions):
    """
    Compute the critical path length (longest path) in the DAG.
    edges: dict mapping position → set of positions it depends on
    Returns: depth (0 = all independent, 1 = one level of deps, etc.)
    """
    # Build adjacency: parent → children
    children = defaultdict(set)
    for pos, deps in edges.items():
        for dep in deps:
            children[dep].add(pos)

    # Compute depth of each node
    depths = {}
    def get_depth(pos):
        if pos in depths:
            return depths[pos]
        deps = edges.get(pos, set())
        if not deps:
            depths[pos] = 0
            return 0
        d = max(get_depth(dep) for dep in deps) + 1
        depths[pos] = d
        return d

    for pos in positions:
        get_depth(pos)

    return max(depths.values()) if depths else 0


def compute_parallel_schedule(edges, positions):
    """
    Compute the optimal parallel generation schedule.
    Returns list of sets: each set contains positions that can be
    generated in parallel at that step.

    Step 0: all positions with no dependencies (roots)
    Step 1: positions whose ALL dependencies were resolved in step 0
    etc.
    """
    # Compute level of each position
    levels = {}
    def get_level(pos):
        if pos in levels:
            return levels[pos]
        deps = edges.get(pos, set())
        if not deps:
            levels[pos] = 0
            return 0
        lev = max(get_level(dep) for dep in deps) + 1
        levels[pos] = lev
        return lev

    for pos in positions:
        get_level(pos)

    # Group by level
    max_level = max(levels.values()) if levels else 0
    schedule = [set() for _ in range(max_level + 1)]
    for pos, lev in levels.items():
        schedule[lev].add(pos)

    return schedule


def generate_dag_sequences(edges, num_values, vocab_size, seq_len, seed=42):
    """
    Generate all valid sequences from a DAG dependency structure.

    Args:
        edges: dict mapping position → set of positions it depends on.
               Positions not in edges are roots (independent).
        num_values: number of distinct values each ROOT position can take.
                    Non-root positions have their value determined by
                    a random function of their dependencies.
        vocab_size: size of shared content vocabulary
        seq_len: total sequence length (including BOS and EOS)
        seed: random seed
    """
    rng = random.Random(seed)
    tok2id, id2tok, V, content_tokens = make_vocab(vocab_size)

    content_positions = list(range(1, seq_len - 1))

    # Identify roots (no dependencies)
    roots = [p for p in content_positions if not edges.get(p, set())]
    non_roots = [p for p in content_positions if edges.get(p, set())]

    # For each root, it can take num_values different values
    # Map: (root_pos, choice_index) → token_id
    root_tokens = {}
    for pos in roots:
        chosen = rng.sample(content_tokens, min(num_values, len(content_tokens)))
        for i, tok in enumerate(chosen):
            root_tokens[(pos, i)] = tok2id[tok]

    # For each non-root, its value is determined by a random function
    # of its dependency values.
    # Map: (pos, tuple_of_dep_values) → token_id
    dep_functions = {}
    for pos in non_roots:
        deps = sorted(edges[pos])
        # Enumerate all possible combinations of dependency values
        dep_value_options = [range(num_values) if d in roots
                           else None for d in deps]

        # We need to know all possible values each dep can take
        # This requires building the table bottom-up
        # For simplicity, we'll generate sequences by enumeration

    # Generate all sequences by enumerating root choices
    # and computing dependent values
    all_root_combos = list(product(range(num_values), repeat=len(roots)))

    # Build dependency resolution order (topological sort)
    schedule = compute_parallel_schedule(edges, content_positions)

    # For non-roots: create random mapping from parent values to token
    # We need to map (position, tuple_of_parent_token_ids) → token_id
    dep_token_map = {}

    sequences = []
    for root_combo in all_root_combos:
        seq = [0] * seq_len
        seq[0] = tok2id["<BOS>"]
        seq[-1] = tok2id["<EOS>"]

        # Set root values
        for i, pos in enumerate(roots):
            seq[pos] = root_tokens[(pos, root_combo[i])]

        # Resolve non-roots in topological order
        for level_positions in schedule:
            for pos in sorted(level_positions):
                if pos in roots:
                    continue
                deps = sorted(edges[pos])
                dep_vals = tuple(seq[d] for d in deps)
                key = (pos, dep_vals)

                if key not in dep_token_map:
                    dep_token_map[key] = tok2id[rng.choice(content_tokens)]

                seq[pos] = dep_token_map[key]

        sequences.append(seq)

    # Compute theoretical minimum steps
    depth = compute_dag_depth(edges, content_positions)
    min_steps = depth + 1  # +1 because roots need 1 step too

    return {
        "name": f"dag_d{depth}_v{vocab_size}_s{seed}",
        "description": f"DAG depth={depth}, vocab={vocab_size}, "
                       f"{len(roots)} roots, {len(non_roots)} deps",
        "sequential_depth": depth,
        "min_steps": min_steps,
        "tok2id": tok2id, "id2tok": id2tok, "V": V,
        "sequences": sequences,
        "seq_len": seq_len,
        "content_positions": content_positions,
        "edges": edges,
        "schedule": schedule,
        "roots": roots,
        "non_roots": non_roots,
    }


# ═══════════════════════════════════════════════════════════════════
# Predefined DAG structures
# ═══════════════════════════════════════════════════════════════════

def dag_independent(seed=42):
    """
    All 4 positions independent. Depth=0.
         1   2   3   4
    (no edges)
    Optimal: 1 step
    """
    return generate_dag_sequences(
        edges={},
        num_values=3, vocab_size=6, seq_len=6, seed=seed,
    )


def dag_chain(seed=42):
    """
    Pure chain: 1→2→3→4. Depth=3.
    Fully sequential — no parallelism possible.
    Optimal: 4 steps
    """
    return generate_dag_sequences(
        edges={2: {1}, 3: {2}, 4: {3}},
        num_values=2, vocab_size=6, seq_len=6, seed=seed,
    )


def dag_tree_wide(seed=42):
    """
    Wide tree: 1→{2,3,4,5}. Depth=1.
    One decision, then 4 parallel tokens.
    Optimal: 2 steps
    """
    return generate_dag_sequences(
        edges={2: {1}, 3: {1}, 4: {1}, 5: {1}},
        num_values=3, vocab_size=6, seq_len=7, seed=seed,
    )


def dag_diamond(seed=42):
    """
    Diamond: 1→{2,3}→4. Depth=2.
        1
       / \
      2   3
       \ /
        4
    Token 4 depends on BOTH 2 and 3.
    Optimal: 3 steps (resolve 1, then 2&3 parallel, then 4)
    """
    return generate_dag_sequences(
        edges={2: {1}, 3: {1}, 4: {2, 3}},
        num_values=2, vocab_size=6, seq_len=6, seed=seed,
    )


def dag_wide_diamond(seed=42):
    """
    Wide diamond: 1→{2,3,4}→5. Depth=2.
        1
      / | \
     2  3  4
      \ | /
        5
    3 parallel middle tokens, then convergence.
    Optimal: 3 steps
    """
    return generate_dag_sequences(
        edges={2: {1}, 3: {1}, 4: {1}, 5: {2, 3, 4}},
        num_values=2, vocab_size=6, seq_len=7, seed=seed,
    )


def dag_two_roots(seed=42):
    """
    Two independent roots, each with dependents:
      1 → 3
      2 → 4
      {3,4} → 5
    Depth=2. Positions 1,2 are parallel roots. 3,4 depend on them.
    5 depends on both 3 and 4.
    Optimal: 3 steps (1&2 parallel, 3&4 parallel, then 5)
    """
    return generate_dag_sequences(
        edges={3: {1}, 4: {2}, 5: {3, 4}},
        num_values=2, vocab_size=6, seq_len=7, seed=seed,
    )


def dag_hourglass(seed=42):
    """
    Hourglass: {1,2}→3→{4,5}. Depth=2.
    Two roots converge to one bottleneck, then fan out.
      1   2
       \ /
        3
       / \
      4   5
    Optimal: 3 steps (1&2, then 3, then 4&5)
    """
    return generate_dag_sequences(
        edges={3: {1, 2}, 4: {3}, 5: {3}},
        num_values=2, vocab_size=6, seq_len=7, seed=seed,
    )


def dag_mixed_depth(seed=42):
    """
    Mixed depths in one sequence:
    Position 1: root (depth 0)
    Position 2: depends on 1 (depth 1)
    Position 3: root (depth 0)
    Position 4: depends on 2 and 3 (depth 2)
    Position 5: depends on 1 (depth 1)

    Critical path: 1→2→4 or 3→4, depth=2
    Schedule: {1,3} parallel, {2,5} parallel, {4}
    Optimal: 3 steps
    """
    return generate_dag_sequences(
        edges={2: {1}, 4: {2, 3}, 5: {1}},
        num_values=2, vocab_size=6, seq_len=7, seed=seed,
    )


def dag_deep_narrow(seed=42):
    """
    Deep narrow: 1→2→3→4→5 pure chain. Depth=4.
    No parallelism at all.
    Optimal: 5 steps
    """
    return generate_dag_sequences(
        edges={2: {1}, 3: {2}, 4: {3}, 5: {4}},
        num_values=2, vocab_size=6, seq_len=7, seed=seed,
    )


ALL_V3_DATASETS = [
    ("Independent (d=0)", dag_independent),
    ("Chain (d=3)", dag_chain),
    ("Wide tree (d=1)", dag_tree_wide),
    ("Diamond (d=2)", dag_diamond),
    ("Wide diamond (d=2)", dag_wide_diamond),
    ("Two roots (d=2)", dag_two_roots),
    ("Hourglass (d=2)", dag_hourglass),
    ("Mixed depth (d=2)", dag_mixed_depth),
    ("Deep narrow (d=4)", dag_deep_narrow),
]


if __name__ == "__main__":
    print("V3 Datasets (DAG-based dependencies):\n")

    for label, fn in ALL_V3_DATASETS:
        ds = fn()
        schedule = ds["schedule"]
        sched_str = " → ".join(
            "{" + ",".join(str(p) for p in sorted(s)) + "}" for s in schedule
        )

        print(f"  {label}")
        print(f"    {ds['description']}")
        print(f"    seqs={len(ds['sequences']):3d}  len={ds['seq_len']}  V={ds['V']}  "
              f"depth={ds['sequential_depth']}  min_steps={ds['min_steps']}")
        print(f"    Optimal schedule: {sched_str}")

        # Show edges
        edge_strs = []
        for pos in sorted(ds["edges"].keys()):
            deps = sorted(ds["edges"][pos])
            edge_strs.append(f"{pos}←{{{','.join(str(d) for d in deps)}}}")
        if edge_strs:
            print(f"    Dependencies: {', '.join(edge_strs)}")
        else:
            print(f"    Dependencies: none (fully independent)")

        # Show token reuse per position
        for pos in ds["content_positions"]:
            unique = set(seq[pos] for seq in ds["sequences"])
            toks = [ds["id2tok"][t] for t in sorted(unique)]
            print(f"    pos {pos}: {len(unique)} values ({', '.join(toks)})")

        # Show first few sequences
        for seq in ds["sequences"][:3]:
            tokens = [ds["id2tok"][t] for t in seq]
            print(f"      {' '.join(tokens)}")
        if len(ds["sequences"]) > 3:
            print(f"      ... ({len(ds['sequences']) - 3} more)")
        print()

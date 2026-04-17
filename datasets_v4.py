"""
Datasets v4: Large randomly-generated DAG datasets.

Key improvements:
- Much larger: configurable branching factor generates 16-256+ sequences
- Random DAG generation: not hand-crafted structures
- Shared vocabulary makes memorization harder
- Visualization of the dependency DAGs
"""

import torch
import random
import math
from collections import defaultdict
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os


def make_vocab(vocab_size):
    special = ["<BOS>", "<EOS>", "<MASK>"]
    content_tokens = [f"t{i}" for i in range(vocab_size)]
    all_tokens = special + content_tokens
    tok2id = {t: i for i, t in enumerate(all_tokens)}
    id2tok = {i: t for t, i in tok2id.items()}
    return tok2id, id2tok, len(all_tokens), content_tokens


def compute_dag_levels(edges, positions):
    """Compute level of each position (0 = root)."""
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
    return levels


def compute_schedule(edges, positions):
    """Optimal parallel schedule: list of sets."""
    levels = compute_dag_levels(edges, positions)
    max_level = max(levels.values()) if levels else 0
    schedule = [set() for _ in range(max_level + 1)]
    for pos, lev in levels.items():
        schedule[lev].add(pos)
    return schedule


def generate_random_dag(num_positions, edge_prob=0.3, max_parents=2, seed=42):
    """
    Generate a random DAG over positions 1..num_positions.
    Each position can depend on earlier positions with probability edge_prob.
    """
    rng = random.Random(seed)
    edges = {}
    for pos in range(2, num_positions + 1):
        # Can depend on any earlier position
        possible_parents = list(range(1, pos))
        parents = set()
        for p in possible_parents:
            if rng.random() < edge_prob and len(parents) < max_parents:
                parents.add(p)
        if parents:
            edges[pos] = parents
    return edges


def generate_dag_dataset(edges, num_positions, values_per_root=3,
                          vocab_size=8, seed=42):
    """
    Generate sequences from a DAG.
    Root positions: each independently takes values_per_root values.
    Non-root positions: value is a random function of parent values.
    """
    rng = random.Random(seed)
    tok2id, id2tok, V, content_tokens = make_vocab(vocab_size)
    seq_len = num_positions + 2  # BOS + content + EOS
    content_pos = list(range(1, num_positions + 1))

    roots = [p for p in content_pos if not edges.get(p, set())]
    non_roots = [p for p in content_pos if edges.get(p, set())]

    # Assign random tokens to root values
    root_tokens = {}
    for pos in roots:
        chosen = [tok2id[rng.choice(content_tokens)]
                  for _ in range(values_per_root)]
        for i, tok in enumerate(chosen):
            root_tokens[(pos, i)] = tok

    # Generate all root combinations
    num_seqs = values_per_root ** len(roots)
    root_combos = list(product(range(values_per_root), repeat=len(roots)))

    # For non-roots: random function from parent token values to token
    schedule = compute_schedule(edges, content_pos)
    dep_map = {}

    sequences = []
    for combo in root_combos:
        seq = [0] * seq_len
        seq[0] = tok2id["<BOS>"]
        seq[-1] = tok2id["<EOS>"]

        # Set roots
        for i, pos in enumerate(roots):
            seq[pos] = root_tokens[(pos, combo[i])]

        # Fill non-roots in topological order
        for level_set in schedule:
            for pos in sorted(level_set):
                if pos in roots:
                    continue
                deps = sorted(edges[pos])
                dep_vals = tuple(seq[d] for d in deps)
                key = (pos, dep_vals)
                if key not in dep_map:
                    dep_map[key] = tok2id[rng.choice(content_tokens)]
                seq[pos] = dep_map[key]

        sequences.append(seq)

    # Deduplicate (shared vocab can create duplicates)
    unique_seqs = []
    seen = set()
    for seq in sequences:
        t = tuple(seq)
        if t not in seen:
            seen.add(t)
            unique_seqs.append(seq)

    depth = max(compute_dag_levels(edges, content_pos).values()) if non_roots else 0
    min_steps = depth + 1

    return {
        "name": f"dag_n{num_positions}_d{depth}_v{vocab_size}_s{seed}",
        "description": (f"Random DAG: {num_positions} positions, depth={depth}, "
                        f"vocab={vocab_size}, {len(roots)} roots, "
                        f"{len(unique_seqs)} unique seqs"),
        "sequential_depth": depth,
        "min_steps": min_steps,
        "tok2id": tok2id, "id2tok": id2tok, "V": V,
        "sequences": unique_seqs,
        "seq_len": seq_len,
        "content_positions": content_pos,
        "edges": edges,
        "schedule": schedule,
        "roots": roots,
        "non_roots": non_roots,
    }


# ═══════════════════════════════════════════════════════════════════
# Predefined large datasets
# ═══════════════════════════════════════════════════════════════════

def large_independent(seed=42):
    """6 positions, all independent, 3 values each = 729 sequences."""
    return generate_dag_dataset(
        edges={}, num_positions=6, values_per_root=3,
        vocab_size=8, seed=seed,
    )


def large_wide_tree(seed=42):
    """{1,2} roots → {3,4,5,6} depend on both roots."""
    edges = {3: {1}, 4: {1}, 5: {2}, 6: {2}}
    return generate_dag_dataset(
        edges=edges, num_positions=6, values_per_root=4,
        vocab_size=8, seed=seed,
    )


def large_chain(seed=42):
    """Two independent chains: 1→2→3, 4→5→6. 2 roots × 4 values = 16 seqs."""
    edges = {2: {1}, 3: {2}, 5: {4}, 6: {5}}
    return generate_dag_dataset(
        edges=edges, num_positions=6, values_per_root=4,
        vocab_size=8, seed=seed,
    )


def large_diamond(seed=42):
    """{1,2}→{3,4}→5→{6,7}. 2 roots, diamond + fan."""
    edges = {3: {1}, 4: {2}, 5: {3, 4}, 6: {5}, 7: {5}}
    return generate_dag_dataset(
        edges=edges, num_positions=7, values_per_root=4,
        vocab_size=8, seed=seed,
    )


def large_random_sparse(seed=42):
    """Random DAG with low edge probability."""
    edges = generate_random_dag(8, edge_prob=0.2, max_parents=2, seed=seed)
    return generate_dag_dataset(
        edges=edges, num_positions=8, values_per_root=3,
        vocab_size=10, seed=seed,
    )


def large_random_dense(seed=42):
    """Random DAG with high edge probability."""
    edges = generate_random_dag(8, edge_prob=0.5, max_parents=2, seed=seed)
    return generate_dag_dataset(
        edges=edges, num_positions=8, values_per_root=3,
        vocab_size=10, seed=seed,
    )


def large_hourglass(seed=42):
    """{1,2,3}→4→{5,6,7}. Wide-narrow-wide. 3 roots."""
    edges = {4: {1, 2, 3}, 5: {4}, 6: {4}, 7: {4}}
    return generate_dag_dataset(
        edges=edges, num_positions=7, values_per_root=3,
        vocab_size=8, seed=seed,
    )


ALL_V4_DATASETS = [
    ("Independent (6pos)", large_independent),
    ("Wide tree (1→5)", large_wide_tree),
    ("Chain (6)", large_chain),
    ("Diamond chain", large_diamond),
    ("Random sparse", large_random_sparse),
    ("Random dense", large_random_dense),
    ("Hourglass", large_hourglass),
]


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

C_ROOT = "#E74C3C"
C_DEP = "#3498DB"
C_BOS = "#888888"


def visualize_dag(ds, ax):
    """Draw the DAG structure for a dataset."""
    edges = ds["edges"]
    content_pos = ds["content_positions"]
    schedule = ds["schedule"]
    levels = compute_dag_levels(edges, content_pos)
    roots = ds["roots"]

    ax.set_aspect("equal")
    ax.axis("off")

    # Layout: levels on y-axis, positions spread on x-axis
    max_level = max(levels.values()) if levels else 0
    level_groups = defaultdict(list)
    for pos in content_pos:
        level_groups[levels[pos]].append(pos)

    pos_coords = {}
    for lev, positions in level_groups.items():
        n = len(positions)
        for i, pos in enumerate(sorted(positions)):
            x = (i - (n - 1) / 2) * 1.5
            y = (max_level - lev) * 1.5
            pos_coords[pos] = (x, y)

    # Draw edges first
    for pos, deps in edges.items():
        if pos in pos_coords:
            for dep in deps:
                if dep in pos_coords:
                    x1, y1 = pos_coords[dep]
                    x2, y2 = pos_coords[pos]
                    ax.annotate("", xy=(x2, y2 + 0.3), xytext=(x1, y1 - 0.3),
                                arrowprops=dict(arrowstyle="->", lw=1.5,
                                                color="#555", connectionstyle="arc3,rad=0.1"))

    # Draw nodes
    for pos in content_pos:
        x, y = pos_coords[pos]
        color = C_ROOT if pos in roots else C_DEP
        n_unique = len(set(seq[pos] for seq in ds["sequences"]))
        ax.add_patch(FancyBboxPatch((x - 0.4, y - 0.3), 0.8, 0.6,
                                     boxstyle="round,pad=0.08",
                                     facecolor=color, edgecolor="black",
                                     linewidth=1.5))
        ax.text(x, y, f"p{pos}", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        ax.text(x, y - 0.5, f"{n_unique}v", ha="center", va="center",
                fontsize=7, color="#666")

    # Schedule text
    sched_str = " → ".join(
        "{" + ",".join(f"p{p}" for p in sorted(s)) + "}" for s in schedule
    )

    ax.set_title(f"{ds['name']}\n{len(ds['sequences'])} seqs, depth={ds['sequential_depth']}, "
                 f"min_steps={ds['min_steps']}\n{sched_str}",
                 fontsize=9, fontweight="bold", pad=5)

    # Set limits
    if pos_coords:
        xs = [c[0] for c in pos_coords.values()]
        ys = [c[1] for c in pos_coords.values()]
        margin = 1.0
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)


def visualize_all_dags():
    """Create a figure showing all DAG structures."""
    os.makedirs("figures", exist_ok=True)

    n = len(ALL_V4_DATASETS)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, (label, fn) in enumerate(ALL_V4_DATASETS):
        ds = fn()
        visualize_dag(ds, axes[i])

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # Legend
    legend_items = [
        mpatches.Patch(facecolor=C_ROOT, edgecolor="black", label="Root (independent)"),
        mpatches.Patch(facecolor=C_DEP, edgecolor="black", label="Dependent"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("V4 Dataset DAG Structures\n(node labels: position, Nv = unique values)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/v4_dag_structures.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    print("Saved figures/v4_dag_structures.png")
    plt.close()


if __name__ == "__main__":
    print("V4 Datasets (large, randomly generated):\n")
    for label, fn in ALL_V4_DATASETS:
        ds = fn()
        sched = ds["schedule"]
        sched_str = " → ".join(
            "{" + ",".join(f"p{p}" for p in sorted(s)) + "}" for s in sched
        )
        print(f"  {label}")
        print(f"    {ds['description']}")
        print(f"    seqs={len(ds['sequences']):4d}  len={ds['seq_len']}  V={ds['V']}  "
              f"depth={ds['sequential_depth']}  min_steps={ds['min_steps']}")
        print(f"    Schedule: {sched_str}")

        # Edge info
        edge_strs = []
        for pos in sorted(ds["edges"].keys()):
            deps = sorted(ds["edges"][pos])
            edge_strs.append(f"p{pos}←{{{','.join(f'p{d}' for d in deps)}}}")
        print(f"    Deps: {', '.join(edge_strs) if edge_strs else 'none'}")

        # Sample sequences
        for seq in ds["sequences"][:3]:
            tokens = [ds["id2tok"][t] for t in seq]
            print(f"      {' '.join(tokens)}")
        if len(ds["sequences"]) > 3:
            print(f"      ... ({len(ds['sequences']) - 3} more)")
        print()

    # Visualize
    visualize_all_dags()

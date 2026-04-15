"""
Visualizations for Parallel Generation experiments.

1. Dependency trees for each dataset
2. Step-by-step resolution traces for each method
3. Summary comparison charts
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# Colors
# ═══════════════════════════════════════════════════════════════════
C_BOS = "#888888"
C_EOS = "#888888"
C_BRANCH = "#E74C3C"      # red — branching decision (high entropy)
C_DETERMINED = "#2ECC71"   # green — determined once branch is known
C_UNKNOWN = "#F39C12"      # orange — not yet resolved
C_MASK = "#BDC3C7"         # gray — masked
C_RESOLVED = "#3498DB"     # blue — just resolved this step


# ═══════════════════════════════════════════════════════════════════
# Figure 1: Dependency Trees
# ═══════════════════════════════════════════════════════════════════

def draw_tree_depth1_wide(ax):
    """depth1_wide: 1 branch → 4 parallel leaves"""
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("depth1_wide: 1 branch → 4 parallel leaves\n(Sequential depth = 1)",
                 fontsize=13, fontweight="bold", pad=10)

    # BOS
    ax.add_patch(FancyBboxPatch((0, 2.7), 1.2, 0.6, boxstyle="round,pad=0.1",
                                 facecolor=C_BOS, edgecolor="black", linewidth=1.5))
    ax.text(0.6, 3.0, "BOS", ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    # Branch node
    ax.add_patch(FancyBboxPatch((2.2, 2.7), 1.6, 0.6, boxstyle="round,pad=0.1",
                                 facecolor=C_BRANCH, edgecolor="black", linewidth=1.5))
    ax.text(3.0, 3.0, "A1|A2", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    ax.annotate("", xy=(2.2, 3.0), xytext=(1.2, 3.0),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))

    # Entropy label
    ax.text(3.0, 2.3, "H = 1 bit", ha="center", va="center", fontsize=9,
            color=C_BRANCH, fontstyle="italic")

    # Parallel leaves
    leaf_labels = ["B", "C", "D", "E"]
    for i, label in enumerate(leaf_labels):
        y = 4.2 - i * 1.1
        ax.add_patch(FancyBboxPatch((5.2, y - 0.3), 1.6, 0.6, boxstyle="round,pad=0.1",
                                     facecolor=C_DETERMINED, edgecolor="black", linewidth=1.5))
        ax.text(6.0, y, f"{label}1|{label}2", ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")
        ax.annotate("", xy=(5.2, y), xytext=(3.8, 3.0),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
        ax.text(7.1, y, "H = 0 | A", ha="left", va="center", fontsize=9,
                color=C_DETERMINED, fontstyle="italic")

    # EOS
    ax.add_patch(FancyBboxPatch((8.5, 2.7), 1.2, 0.6, boxstyle="round,pad=0.1",
                                 facecolor=C_EOS, edgecolor="black", linewidth=1.5))
    ax.text(9.1, 3.0, "EOS", ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    # Legend
    legend_items = [
        mpatches.Patch(facecolor=C_BRANCH, edgecolor="black", label="Branching (high entropy)"),
        mpatches.Patch(facecolor=C_DETERMINED, edgecolor="black", label="Determined (H=0 given parent)"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=9, framealpha=0.9)


def draw_tree_depth2(ax):
    """depth2: 2 sequential branches"""
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("depth2: 2 sequential branches → 1 determined leaf\n(Sequential depth = 2)",
                 fontsize=13, fontweight="bold", pad=10)

    # BOS
    ax.add_patch(FancyBboxPatch((0, 2.2), 1.0, 0.6, boxstyle="round,pad=0.1",
                                 facecolor=C_BOS, edgecolor="black", linewidth=1.5))
    ax.text(0.5, 2.5, "BOS", ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    # Level 1: first branch
    ax.add_patch(FancyBboxPatch((2.0, 2.2), 1.2, 0.6, boxstyle="round,pad=0.1",
                                 facecolor=C_BRANCH, edgecolor="black", linewidth=1.5))
    ax.text(2.6, 2.5, "A1|A2", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    ax.annotate("", xy=(2.0, 2.5), xytext=(1.0, 2.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.text(2.6, 1.7, "H = 1 bit", ha="center", va="center", fontsize=9,
            color=C_BRANCH, fontstyle="italic")

    # Level 2: second branch (depends on first)
    for i, (y, label) in enumerate([(3.7, "B1|B2"), (1.3, "B3|B4")]):
        ax.add_patch(FancyBboxPatch((4.5, y - 0.3), 1.2, 0.6, boxstyle="round,pad=0.1",
                                     facecolor=C_BRANCH, edgecolor="black", linewidth=1.5))
        ax.text(5.1, y, label, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.annotate("", xy=(4.5, y), xytext=(3.2, 2.5),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

    ax.text(6.0, 1.0, "H = 1 | A", ha="center", va="center", fontsize=9,
            color=C_BRANCH, fontstyle="italic")

    # Level 3: determined leaves
    leaf_ys = [4.2, 3.2, 1.8, 0.8]
    leaf_labels = ["C1", "C2", "C3", "C4"]
    parent_ys = [3.7, 3.7, 1.3, 1.3]
    for y, label, py in zip(leaf_ys, leaf_labels, parent_ys):
        ax.add_patch(FancyBboxPatch((7.0, y - 0.3), 1.0, 0.6, boxstyle="round,pad=0.1",
                                     facecolor=C_DETERMINED, edgecolor="black", linewidth=1.5))
        ax.text(7.5, y, label, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.annotate("", xy=(7.0, y), xytext=(5.7, py),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

    # EOS
    ax.add_patch(FancyBboxPatch((8.7, 2.2), 1.0, 0.6, boxstyle="round,pad=0.1",
                                 facecolor=C_EOS, edgecolor="black", linewidth=1.5))
    ax.text(9.2, 2.5, "EOS", ha="center", va="center", fontsize=10, color="white", fontweight="bold")


def draw_tree_independent(ax):
    """independent: all tokens independent"""
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("independent: All tokens independent\n(Sequential depth = 0)",
                 fontsize=13, fontweight="bold", pad=10)

    # BOS
    ax.add_patch(FancyBboxPatch((0, 1.7), 1.0, 0.6, boxstyle="round,pad=0.1",
                                 facecolor=C_BOS, edgecolor="black", linewidth=1.5))
    ax.text(0.5, 2.0, "BOS", ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    # Three independent tokens
    labels = ["A1|A2", "B1|B2", "C1|C2"]
    for i, label in enumerate(labels):
        y = 3.2 - i * 1.2
        ax.add_patch(FancyBboxPatch((3.0, y - 0.3), 1.4, 0.6, boxstyle="round,pad=0.1",
                                     facecolor=C_DETERMINED, edgecolor="black", linewidth=1.5))
        ax.text(3.7, y, label, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.annotate("", xy=(3.0, y), xytext=(1.0, 2.0),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="black", linestyle="--"))
        ax.text(4.7, y, "H = 1 bit (indep.)", ha="left", va="center", fontsize=9,
                color="#27AE60", fontstyle="italic")

    # EOS
    ax.add_patch(FancyBboxPatch((7.5, 1.7), 1.0, 0.6, boxstyle="round,pad=0.1",
                                 facecolor=C_EOS, edgecolor="black", linewidth=1.5))
    ax.text(8.0, 2.0, "EOS", ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    ax.text(5.0, -0.2, "No dependencies — all can be resolved in 1 parallel step",
            ha="center", fontsize=10, fontstyle="italic", color="#555555")


def fig_dependency_trees():
    fig, axes = plt.subplots(1, 3, figsize=(30, 6))
    draw_tree_independent(axes[0])
    draw_tree_depth1_wide(axes[1])
    draw_tree_depth2(axes[2])
    fig.suptitle("Dependency Structures of Token Sequences", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/dependency_trees.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved {OUT_DIR}/dependency_trees.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Figure 2: Step-by-step resolution traces
# ═══════════════════════════════════════════════════════════════════

def draw_resolution_trace(ax, title, steps, seq_labels):
    """
    steps: list of lists — each inner list is the state at that step.
         state is list of (token_label, color) per position.
    seq_labels: list of position labels (e.g. ["BOS", "pos1", "pos2", ..., "EOS"])
    """
    n_steps = len(steps)
    n_pos = len(seq_labels)

    ax.set_xlim(-0.5, n_pos - 0.5)
    ax.set_ylim(-0.5, n_steps - 0.3)
    ax.set_aspect(0.8)
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    for step_idx, state in enumerate(steps):
        y = n_steps - 1 - step_idx
        for pos_idx, (label, color) in enumerate(state):
            x = pos_idx
            ax.add_patch(FancyBboxPatch((x - 0.4, y - 0.25), 0.8, 0.5,
                                         boxstyle="round,pad=0.05",
                                         facecolor=color, edgecolor="black", linewidth=1))
            fontsize = 7 if len(label) > 3 else 8
            ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
                    color="white" if color not in [C_MASK, C_UNKNOWN] else "black",
                    fontweight="bold")

        # Step label on the left
        ax.text(-0.7, y, f"t={step_idx}", ha="right", va="center", fontsize=9, color="#555")

    # Position labels at bottom
    for i, label in enumerate(seq_labels):
        ax.text(i, -0.6, label, ha="center", va="center", fontsize=8, color="#555")


def fig_resolution_traces():
    """Show how AR, Mask-Predict Adaptive, and Diffusion resolve depth1_wide."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    pos_labels = ["BOS", "pos1", "pos2", "pos3", "pos4", "pos5", "EOS"]

    # AR: 5 steps, left to right
    ar_steps = [
        [("BOS", C_BOS), ("?", C_UNKNOWN), ("?", C_UNKNOWN), ("?", C_UNKNOWN), ("?", C_UNKNOWN), ("?", C_UNKNOWN), ("EOS", C_EOS)],
        [("BOS", C_BOS), ("A1", C_RESOLVED), ("?", C_UNKNOWN), ("?", C_UNKNOWN), ("?", C_UNKNOWN), ("?", C_UNKNOWN), ("EOS", C_EOS)],
        [("BOS", C_BOS), ("A1", C_DETERMINED), ("B1", C_RESOLVED), ("?", C_UNKNOWN), ("?", C_UNKNOWN), ("?", C_UNKNOWN), ("EOS", C_EOS)],
        [("BOS", C_BOS), ("A1", C_DETERMINED), ("B1", C_DETERMINED), ("C1", C_RESOLVED), ("?", C_UNKNOWN), ("?", C_UNKNOWN), ("EOS", C_EOS)],
        [("BOS", C_BOS), ("A1", C_DETERMINED), ("B1", C_DETERMINED), ("C1", C_DETERMINED), ("D1", C_RESOLVED), ("?", C_UNKNOWN), ("EOS", C_EOS)],
        [("BOS", C_BOS), ("A1", C_DETERMINED), ("B1", C_DETERMINED), ("C1", C_DETERMINED), ("D1", C_DETERMINED), ("E1", C_RESOLVED), ("EOS", C_EOS)],
    ]
    draw_resolution_trace(axes[0], "Autoregressive\n(5 steps — no parallelism)", ar_steps, pos_labels)

    # Naive Parallel: 1 step but wrong
    naive_steps = [
        [("BOS", C_BOS), ("?", C_UNKNOWN), ("?", C_UNKNOWN), ("?", C_UNKNOWN), ("?", C_UNKNOWN), ("?", C_UNKNOWN), ("EOS", C_EOS)],
        [("BOS", C_BOS), ("A1", C_RESOLVED), ("A2", C_BRANCH), ("A1", C_BRANCH), ("A2", C_BRANCH), ("A1", C_BRANCH), ("EOS", C_EOS)],
    ]
    draw_resolution_trace(axes[1], "Naive Parallel\n(1 step — INVALID)", naive_steps, pos_labels)

    # Mask-Predict Adaptive: 2 steps
    mp_steps = [
        [("BOS", C_BOS), ("M", C_MASK), ("M", C_MASK), ("M", C_MASK), ("M", C_MASK), ("M", C_MASK), ("EOS", C_EOS)],
        [("BOS", C_BOS), ("A1", C_RESOLVED), ("M", C_MASK), ("M", C_MASK), ("M", C_MASK), ("M", C_MASK), ("EOS", C_EOS)],
        [("BOS", C_BOS), ("A1", C_DETERMINED), ("B1", C_RESOLVED), ("C1", C_RESOLVED), ("D1", C_RESOLVED), ("E1", C_RESOLVED), ("EOS", C_EOS)],
    ]
    draw_resolution_trace(axes[2], "Mask-Predict Adaptive\n(2 steps — exploits parallelism!)", mp_steps, pos_labels)

    # Diffusion Adaptive: 2 steps (same as MP)
    diff_steps = [
        [("BOS", C_BOS), ("M", C_MASK), ("M", C_MASK), ("M", C_MASK), ("M", C_MASK), ("M", C_MASK), ("EOS", C_EOS)],
        [("BOS", C_BOS), ("A1", C_RESOLVED), ("M", C_MASK), ("M", C_MASK), ("M", C_MASK), ("M", C_MASK), ("EOS", C_EOS)],
        [("BOS", C_BOS), ("A1", C_DETERMINED), ("B1", C_RESOLVED), ("C1", C_RESOLVED), ("D1", C_RESOLVED), ("E1", C_RESOLVED), ("EOS", C_EOS)],
    ]
    draw_resolution_trace(axes[3], "Diffusion Adaptive\n(2 steps — matches theoretical min!)", diff_steps, pos_labels)

    fig.suptitle("Step-by-step Resolution: depth1_wide dataset\n"
                 "1 branching decision (A1|A2) → 4 determined tokens",
                 fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/resolution_traces.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved {OUT_DIR}/resolution_traces.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Figure 3: Summary bar charts
# ═══════════════════════════════════════════════════════════════════

def fig_steps_comparison():
    datasets = ["independent\n(d=0)", "depth1\n(d=1)", "depth1_wide\n(d=1)",
                "depth2\n(d=2)", "depth3\n(d=3)", "mixed\n(d=1-2)"]

    # From benchmark results
    data = {
        "AR":       [3.00, 2.00, 5.00, 3.00, 4.00, 3.00],
        "Jacobi-A": [2.30, 2.42, 4.70, 3.79, 3.77, 3.92],
        "MP-Ad":    [3.00, 2.00, 2.00, 3.00, 4.00, 3.00],
        "Spec":     [3.01, 2.29, 5.39, 3.52, 4.67, 3.60],
        "Diffusion":[3.00, 2.00, 2.00, 3.00, 4.00, 2.00],
    }

    theoretical_min = [1, 2, 2, 3, 4, 2]  # depth + 1, mixed avg

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(datasets))
    width = 0.14
    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6", "#F39C12"]

    for i, (method, values) in enumerate(data.items()):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, values, width, label=method, color=colors[i],
                      edgecolor="black", linewidth=0.5)

    # Theoretical minimum line
    for i, (xpos, ymin) in enumerate(zip(x, theoretical_min)):
        ax.plot([xpos - 2.5*width, xpos + 2.5*width], [ymin, ymin],
                color="black", linewidth=2, linestyle="--", zorder=5)

    # Add one dashed line to legend
    ax.plot([], [], color="black", linewidth=2, linestyle="--", label="Theoretical min (depth+1)")

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Average Steps to Resolve", fontsize=12)
    ax.set_title("Steps to Generate Valid Sequence\n(lower is better)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, 6.5)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/steps_comparison.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved {OUT_DIR}/steps_comparison.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Figure 4: The Trilemma
# ═══════════════════════════════════════════════════════════════════

def fig_trilemma():
    """Scatter: validity vs steps, with diversity as marker size."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Data from depth1_wide (the most revealing dataset)
    methods = {
        "AR":           {"steps": 5.00, "valid": 100.0, "diverse": True},
        "Jacobi-A":     {"steps": 4.70, "valid": 100.0, "diverse": False},
        "Jacobi-S":     {"steps": 17.48, "valid": 30.0,  "diverse": True},
        "MP":           {"steps": 5.00, "valid": 100.0, "diverse": False},
        "MP-Adaptive":  {"steps": 2.00, "valid": 100.0, "diverse": False},
        "Speculative":  {"steps": 5.39, "valid": 100.0, "diverse": True},
        "Diffusion":    {"steps": 2.00, "valid": 100.0, "diverse": False},
    }

    for name, d in methods.items():
        color = "#2ECC71" if d["diverse"] else "#E74C3C"
        marker = "o" if d["diverse"] else "s"
        size = 200 if d["valid"] == 100 else 100
        edge = "black" if d["valid"] == 100 else "red"
        edgewidth = 2 if d["valid"] < 100 else 1.5

        ax.scatter(d["steps"], d["valid"], s=size, c=color, marker=marker,
                   edgecolors=edge, linewidths=edgewidth, zorder=5)

        # Label offset
        xoff, yoff = 0.3, 0
        if name == "Diffusion":
            yoff = -3
        elif name == "MP-Adaptive":
            yoff = 3
        elif name == "AR":
            xoff = -0.3
            yoff = -3
        elif name == "Jacobi-S":
            xoff = -0.5

        ax.annotate(name, (d["steps"] + xoff, d["valid"] + yoff),
                    fontsize=10, fontweight="bold")

    # Theoretical minimum
    ax.axvline(x=2, color="black", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.text(2.1, 50, "Theoretical\nminimum\n(depth+1=2)", fontsize=9,
            color="black", alpha=0.7)

    # Ideal region
    ax.add_patch(plt.Rectangle((1.5, 95), 1.5, 6, facecolor="#2ECC71", alpha=0.15,
                                edgecolor="#2ECC71", linewidth=2, linestyle="--"))
    ax.text(2.25, 96, "IDEAL\n(fast + valid + diverse)", ha="center", fontsize=9,
            color="#27AE60", fontweight="bold")

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#2ECC71", edgecolor="black", label="Diverse (correct distribution)"),
        mpatches.Patch(facecolor="#E74C3C", edgecolor="black", label="Mode collapse (single output)"),
    ]
    ax.legend(handles=legend_items, fontsize=11, loc="lower left")

    ax.set_xlabel("Average Steps to Resolve", fontsize=12)
    ax.set_ylabel("Validity %", fontsize=12)
    ax.set_title("The Validity–Diversity–Speed Trilemma\n(depth1_wide dataset: 5 tokens, depth=1)",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, 20)
    ax.set_ylim(20, 105)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/trilemma.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved {OUT_DIR}/trilemma.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Figure 5: Entropy reduction per step
# ═══════════════════════════════════════════════════════════════════

def fig_entropy_reduction():
    """Show how entropy decreases step-by-step for different methods on depth1_wide."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # depth1_wide: 5 content tokens, 1 branch (1 bit), 4 determined (0 bits each given branch)
    # Total initial entropy: 1 bit (only the branch matters; leaves are deterministic given branch)
    # But from a per-token marginal view: each token has H=1 bit marginally

    # AR: resolves 1 token per step
    # Step 0: 5 tokens unknown. Total H = 1 bit (branch)
    # Step 1: sample A→ H drops to 0 (everything determined). But AR doesn't know this, keeps going.
    # From AR's perspective it resolves tokens left-to-right, each "step" is trivial after step 1.
    ar_h = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # MP-Adaptive / Diffusion:
    # Step 0: all masked, H = 1 bit
    # Step 1: resolve branch (A), H drops to 0
    # Step 2: resolve all 4 leaves in parallel, done
    mp_h = [1.0, 0.0, 0.0]

    ax.plot(range(6), ar_h, "o-", color="#3498DB", linewidth=2.5, markersize=8,
            label="AR (5 steps total)", zorder=5)
    ax.plot(range(3), mp_h, "s-", color="#2ECC71", linewidth=2.5, markersize=8,
            label="MP-Ad / Diffusion (2 steps total)", zorder=5)

    # Shade the "wasted" AR steps
    ax.axvspan(1.5, 5.5, alpha=0.1, color="#E74C3C")
    ax.text(3.5, 0.5, "AR wastes 3 steps\non already-determined tokens",
            ha="center", fontsize=10, color="#E74C3C", fontstyle="italic")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Remaining Entropy (bits)", fontsize=12)
    ax.set_title("Entropy Reduction per Step: depth1_wide\n"
                 "Optimal strategy resolves the highest-entropy token first",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(-0.3, 5.5)
    ax.set_ylim(-0.1, 1.3)
    ax.set_xticks(range(6))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/entropy_reduction.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved {OUT_DIR}/entropy_reduction.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    fig_dependency_trees()
    fig_resolution_traces()
    fig_steps_comparison()
    fig_trilemma()
    fig_entropy_reduction()
    print(f"\nAll figures saved to {OUT_DIR}/")

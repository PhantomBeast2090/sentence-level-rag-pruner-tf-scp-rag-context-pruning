#!/usr/bin/env python3
"""
Publication-Quality Figures for TF-SCP (IEEE Format)
Generates vector PDF plots for:
  1. Context Compression vs. QA Accuracy (F1 vs. Token Reduction)
  2. Impact of Aggressive Thresholding (Twin-axis: Retention & Reduction vs. Beta)
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------------------------------------------------------
# Global Styling Configuration (IEEE Conference / Transactions Standards)
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "Nimbus Roman", "serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

sns.set_theme(style="whitegrid", font="serif", rc={"font.family": "serif"})
sns.set_context("paper", font_scale=1.1)


# -----------------------------------------------------------------------------
# Plot 1: Context Compression vs. QA Accuracy (F1 vs. Token Reduction)
# -----------------------------------------------------------------------------
def plot_compression_vs_f1(output_filename="tf_scp_compression_vs_f1.pdf"):
    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=300)

    # Data definitions
    data = [
        {"name": "Standard RAG\n(Baseline)", "reduction": 0.0, "f1": 50.94, "type": "baseline"},
        {"name": "Hybrid (α=0.5, β=-0.5)", "reduction": 31.20, "f1": 52.70, "type": "hybrid"},
        {"name": "Hybrid (α=0.5, β=0.0)", "reduction": 55.79, "f1": 53.67, "type": "hybrid"},
        {"name": "Hybrid (α=0.75, β=0.0)", "reduction": 60.50, "f1": 54.87, "type": "hybrid"},
        {"name": "BM25-Only (α=1.0, β=0.0)", "reduction": 63.71, "f1": 53.54, "type": "lexical"},
        {"name": "Hybrid (α=0.5, β=+0.5)", "reduction": 74.25, "f1": 57.61, "type": "optimal"},
    ]

    # Color palette & marker mapping
    palette = {
        "baseline": "#4B5563",   # Slate Grey
        "lexical": "#D97706",    # Warm Amber
        "hybrid": "#2563EB",     # Royal Blue
        "optimal": "#DC2626",    # Crimson Red
    }
    markers = {
        "baseline": "s",
        "lexical": "^",
        "hybrid": "o",
        "optimal": "*",
    }

    # Plot baseline reference level
    ax.axhline(50.94, color="#9CA3AF", linestyle=":", linewidth=1.2, zorder=1, label="Standard RAG Baseline (F1=50.94)")

    # Plot points
    for d in data:
        t = d["type"]
        size = 180 if t == "optimal" else (85 if t == "baseline" else 95)
        ax.scatter(
            d["reduction"],
            d["f1"],
            color=palette[t],
            marker=markers[t],
            s=size,
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )

    # Text annotations for individual points to avoid clutter
    # 1. Baseline
    ax.annotate(
        "Standard RAG\n(Full Context)",
        xy=(0.0, 50.94),
        xytext=(3.5, 49.6),
        fontsize=8,
        color="#374151",
        weight="medium",
        ha="left",
    )
    # 2. Hybrid beta = -0.5
    ax.annotate(
        "Hybrid\n(α=0.5, β=-0.5)",
        xy=(31.20, 52.70),
        xytext=(31.20, 51.3),
        fontsize=8,
        color="#1E40AF",
        ha="center",
    )
    # 3. Hybrid beta = 0.0
    ax.annotate(
        "Hybrid\n(α=0.5, β=0.0)",
        xy=(55.79, 53.67),
        xytext=(48.0, 54.6),
        fontsize=8,
        color="#1E40AF",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#93C5FD", lw=0.8),
    )
    # 4. Hybrid alpha = 0.75
    ax.annotate(
        "Hybrid\n(α=0.75, β=0.0)",
        xy=(60.50, 54.87),
        xytext=(60.50, 56.1),
        fontsize=8,
        color="#1E40AF",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#93C5FD", lw=0.8),
    )
    # 5. BM25-Only
    ax.annotate(
        "BM25-Only\n(α=1.0, β=0.0)",
        xy=(63.71, 53.54),
        xytext=(65.0, 52.0),
        fontsize=8,
        color="#B45309",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#FCD34D", lw=0.8),
    )

    # 6. Optimal configuration highlighted
    ax.annotate(
        "Highest F1 & Compression\n(p < 0.001)\nHybrid (α=0.5, β=+0.5)",
        xy=(74.25, 57.61),
        xytext=(50.0, 58.2),
        fontsize=8.5,
        fontweight="bold",
        color="#991B1B",
        ha="right",
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEE2E2", edgecolor="#DC2626", lw=0.9),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#DC2626",
            lw=1.3,
            connectionstyle="arc3,rad=-0.12",
        ),
        zorder=5,
    )

    # Axis properties
    ax.set_xlabel("Token Reduction (%)", fontweight="bold")
    ax.set_ylabel("Token-Level F1 Score", fontweight="bold")
    ax.set_xlim(-5, 85)
    ax.set_ylim(48.5, 60.5)

    # Custom legend
    legend_elements = [
        plt.Line2D([0], [0], marker="s", color="w", label="Standard RAG", markerfacecolor=palette["baseline"], markeredgecolor="k", markersize=7),
        plt.Line2D([0], [0], marker="^", color="w", label="BM25-Only (α=1.0)", markerfacecolor=palette["lexical"], markeredgecolor="k", markersize=7.5),
        plt.Line2D([0], [0], marker="o", color="w", label="TF-SCP Hybrid", markerfacecolor=palette["hybrid"], markeredgecolor="k", markersize=7),
        plt.Line2D([0], [0], marker="*", color="w", label="Optimal TF-SCP (β=+0.5)", markerfacecolor=palette["optimal"], markeredgecolor="k", markersize=10),
    ]
    ax.legend(handles=legend_elements, loc="lower left", frameon=True, framealpha=0.95, edgecolor="#E5E7EB")

    plt.tight_layout()
    plt.savefig(output_filename, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Plot 1: {output_filename}")


# -----------------------------------------------------------------------------
# Plot 2: Impact of Aggressive Thresholding (Twin-axis: Retention & Reduction vs. Beta)
# -----------------------------------------------------------------------------
def plot_threshold_sensitivity(output_filename="tf_scp_threshold_sensitivity.pdf"):
    fig, ax1 = plt.subplots(figsize=(4.4, 3.4), dpi=300)

    # Data
    beta_vals = [-0.5, 0.0, 0.5]
    retention = [98.8, 97.5, 94.5]
    reduction = [31.20, 55.79, 74.25]

    color_ret = "#1D4ED8"   # Solid Royal Blue
    color_red = "#DC2626"   # Dash Crimson Red

    # Y1 (Left): Answer Retention
    line1 = ax1.plot(
        beta_vals,
        retention,
        color=color_ret,
        marker="o",
        markersize=6.5,
        linewidth=1.8,
        label="Answer Retention (%)",
        zorder=3,
    )
    ax1.set_xlabel("Threshold Coefficient (β)", fontweight="bold")
    ax1.set_ylabel("Answer Retention (%)", color=color_ret, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_ret)
    ax1.set_ylim(92.0, 100.5)
    ax1.set_xticks(beta_vals)
    ax1.set_xticklabels(["-0.5 (Permissive)", "0.0 (Mean)", "+0.5 (Aggressive)"], fontsize=8.5)

    # Value labels for Y1
    for x, y in zip(beta_vals, retention):
        ax1.annotate(
            f"{y:.1f}%",
            xy=(x, y),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=color_ret,
        )

    # Y2 (Right): Token Reduction
    ax2 = ax1.twinx()
    ax2.grid(False)  # Avoid overlapping gridlines
    line2 = ax2.plot(
        beta_vals,
        reduction,
        color=color_red,
        marker="s",
        markersize=6,
        linewidth=1.8,
        linestyle="--",
        label="Token Reduction (%)",
        zorder=3,
    )
    ax2.set_ylabel("Token Reduction (%)", color=color_red, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_red)
    ax2.set_ylim(20.0, 85.0)

    # Value labels for Y2
    for x, y in zip(beta_vals, reduction):
        ax2.annotate(
            f"{y:.1f}%",
            xy=(x, y),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=color_red,
        )

    # Unified legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", bbox_to_anchor=(0.04, 0.42), frameon=True, framealpha=0.92, edgecolor="#E5E7EB")

    plt.title("Threshold Sensitivity (TF-SCP, α = 0.5)", pad=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_filename, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Plot 2: {output_filename}")


if __name__ == "__main__":
    plot_compression_vs_f1()
    plot_threshold_sensitivity()

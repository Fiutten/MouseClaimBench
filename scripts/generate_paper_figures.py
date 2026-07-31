"""Generate deterministic vector figures used by the EAAI manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "figures"
PDF_METADATA = {"Creator": "MouseBrainBench", "CreationDate": None, "ModDate": None}


def _box(ax, x: float, y: float, width: float, height: float, title: str, body: str,
         *, facecolor: str = "#f4f6f8", edgecolor: str = "#334155") -> None:
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.15, edgecolor=edgecolor, facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.70, title, ha="center", va="center",
            fontsize=7.7, fontweight="bold", color="#111827")
    ax.text(x + width / 2, y + height * 0.35, body, ha="center", va="center",
            fontsize=6.6, color="#374151", linespacing=1.20)


def _arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                linewidth=1.1, color="#475569"))


def build_framework_workflow() -> None:
    """Render the claim-aware evaluation workflow and its decision boundary."""
    fig, ax = plt.subplots(figsize=(7.4, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    columns = [0.025, 0.27, 0.515, 0.76]
    width = 0.205
    top = 0.47
    height = 0.36
    _box(ax, columns[0], top, width, height, "1. CLAIM CONTRACT",
         "Claim class\nTarget entity and scale\nPermitted interpretation",
         facecolor="#e8f1f8", edgecolor="#315f7d")
    _box(ax, columns[1], top, width, height, "2. EVIDENCE DESIGN",
         "Observable quantities\nBlocks and thresholds\nNulls and hold-outs",
         facecolor="#eef2f7")
    _box(ax, columns[2], top, width, height, "3. EXECUTION",
         "Dataset/model adapters\nMetrics and controls\nConjunctive block gates",
         facecolor="#e9f5ef", edgecolor="#39705a")
    _box(ax, columns[3], top, width, height, "4. DECISION",
         "Admitted claims\nBlocked claims\nVersioned provenance",
         facecolor="#f3f0e8", edgecolor="#796b40")
    for left in columns[:-1]:
        _arrow(ax, (left + width, top + height / 2),
               (left + width + 0.04, top + height / 2))

    _box(ax, 0.085, 0.06, 0.34, 0.22, "ADMITTED",
         "Every required evidence block passes\nfor the declared scope",
         facecolor="#e4f3e9", edgecolor="#26734d")
    _box(ax, 0.575, 0.06, 0.34, 0.22, "BLOCKED OR PARTIAL",
         "Failed or unobserved blocks remain\nexplicit; scores cannot compensate",
         facecolor="#faeceb", edgecolor="#a53b32")
    _arrow(ax, (columns[3] + width / 2, top), (0.745, 0.28))
    _arrow(ax, (columns[3] + width / 2, top), (0.255, 0.28))

    ax.text(0.5, 0.985, "Claim-aware evaluation architecture",
            ha="center", va="top", fontsize=10.5, fontweight="bold", color="#111827")
    ax.text(0.5, 0.90,
            "Models and datasets enter through adapters; interpretation is controlled by the declared evidence contract.",
            ha="center", va="top", fontsize=7.3, color="#4b5563")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "framework_workflow.pdf", bbox_inches="tight", pad_inches=0.05,
                metadata=PDF_METADATA)
    fig.savefig(OUTPUT / "framework_workflow.png", dpi=240, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _annotated_binary_matrix(ax, values: np.ndarray, xlabels: list[str],
                             ylabels: list[str], title: str) -> None:
    """Draw a compact pass/fail matrix with labels suitable for print."""
    cmap = plt.matplotlib.colors.ListedColormap(["#f3d8d5", "#dcefe3"])
    ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=7)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=7)
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=7)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, "PASS" if values[row, col] else "FAIL",
                    ha="center", va="center", fontsize=6.4, fontweight="bold",
                    color="#24563f" if values[row, col] else "#8d2f28")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color("#667085")
        spine.set_linewidth(0.7)


def build_sensitivity_ablation() -> None:
    """Render threshold sensitivity and model-comparison ablations from artifacts."""
    source = ROOT / "results" / "q1_sensitivity" / "summary.json"
    payload = json.loads(source.read_text(encoding="utf-8"))

    allen_rows = payload["allen_vbn"]["rows"]
    allen_blocks = ["Reproducibility", "Topology", "Direction"]
    allen = np.array([
        [block.lower() in row["passed_blocks"] for row in allen_rows]
        for block in allen_blocks
    ], dtype=int)

    static_rows = payload["sensorium_static"]["rows"]
    reliability = sorted({row["reliability_threshold"] for row in static_rows})
    topography = sorted({row["topographic_threshold"] for row in static_rows})
    static = np.array([
        [int(next(row["partial_positive"] for row in static_rows
                  if row["reliability_threshold"] == rel
                  and row["topographic_threshold"] == topo))
         for topo in topography]
        for rel in reliability
    ])

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.55),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.45]})
    _annotated_binary_matrix(
        axes[0], allen,
        [f"{row['threshold_multiplier']:.2f}x" for row in allen_rows],
        allen_blocks,
        "(a) Allen threshold sensitivity",
    )
    axes[0].set_xlabel("Criterion multiplier", fontsize=7)

    _annotated_binary_matrix(
        axes[1], static,
        [f"{value:.2f}" for value in topography],
        [f"{value:.2f}" for value in reliability],
        "(b) Static Sensorium gate",
    )
    axes[1].set_xlabel("Topographic threshold", fontsize=7)
    axes[1].set_ylabel("Reliability threshold", fontsize=7)

    colors = {"mean": "#2f6b8a", "svd": "#a33a32"}
    for cohort_index, cohort in enumerate(payload["dynamic_sensorium"]["cohorts"]):
        rows = cohort["threshold_rows"]
        x = [row["effect_threshold"] for row in rows]
        style = "-" if cohort_index == 0 else "--"
        cohort_name = "Current" if cohort_index == 0 else "OOD"
        axes[2].plot(x, [row["mlp_beats_mean"] for row in rows], style,
                     color=colors["mean"], marker="o", markersize=3,
                     linewidth=1.25, label=f"{cohort_name}: MLP > mean")
        axes[2].plot(x, [row["mlp_beats_svd"] for row in rows], style,
                     color=colors["svd"], marker="s", markersize=3,
                     linewidth=1.25, label=f"{cohort_name}: MLP > SVD")
    axes[2].set_title("(c) Dynamic Sensorium ablation", fontsize=8.5,
                      fontweight="bold", pad=7)
    axes[2].set_xlabel("Required correlation gain", fontsize=7)
    axes[2].set_ylabel("Mice passing (of 5)", fontsize=7)
    axes[2].set_xticks([0.0, 0.002, 0.005, 0.01], ["0", ".002", ".005", ".010"],
                      fontsize=7)
    axes[2].set_yticks(range(0, 6))
    axes[2].tick_params(axis="y", labelsize=7)
    axes[2].set_ylim(-0.15, 5.25)
    axes[2].grid(axis="y", color="#d0d5dd", linewidth=0.6)
    axes[2].legend(loc="lower left", fontsize=5.8, frameon=False, ncol=2)
    for side in ("top", "right"):
        axes[2].spines[side].set_visible(False)

    fig.subplots_adjust(left=0.075, right=0.995, top=0.84, bottom=0.22, wspace=0.48)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "sensitivity_ablation.pdf", bbox_inches="tight", pad_inches=0.03,
                metadata=PDF_METADATA)
    fig.savefig(OUTPUT / "sensitivity_ablation.png", dpi=260,
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    build_framework_workflow()
    build_sensitivity_ablation()

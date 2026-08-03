#!/usr/bin/env python3
"""Build the five data-driven figures for the standards/prospective paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch

from mousebrainbench.knowledge import load_authorization_profile_v2

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
ATTACKS = ROOT / "results/profile_v2_scalability_ablation/summary.json"
DANDI = ROOT / "results/dandi_profile_v2_1/summary.json"

INK = "#20252b"
BLUE = "#2368a2"
TEAL = "#238b78"
AMBER = "#d18f22"
RED = "#b84a4a"
GRAY = "#76808a"
LIGHT = "#edf1f4"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        FIGURES / f"{name}.png",
        dpi=350,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def workflow() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    stages = [
        (0.25, "1. Evidence", "Artifacts\nPredicates\nObservations\nAttestations", BLUE),
        (3.20, "2. Structure", "RDF / PROV-O\nExternal SHACL\nreport", TEAL),
        (6.15, "3. Integrity", "Hashes and lineage\nCycles and conflicts\nIndependence", AMBER),
        (9.10, "4. Authorization", "Required evidence\nComplete deficits\nBounded decision", RED),
    ]
    for index, (x, title, body, color) in enumerate(stages):
        box = FancyBboxPatch(
            (x, 1.15),
            2.55,
            1.75,
            boxstyle="round,pad=0.03,rounding_size=0.07",
            linewidth=1.2,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(box)
        ax.add_patch(
            FancyBboxPatch(
                (x, 2.55),
                2.55,
                0.35,
                boxstyle="round,pad=0.03,rounding_size=0.07",
                linewidth=0,
                facecolor=color,
            )
        )
        ax.text(
            x + 0.14,
            2.72,
            title,
            color="white",
            weight="bold",
            va="center",
            fontsize=11.5,
        )
        ax.text(
            x + 1.275,
            1.87,
            body,
            ha="center",
            va="center",
            color=INK,
            linespacing=1.25,
            fontsize=10.5,
        )
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + 2.93, 2.02),
                xytext=(x + 2.60, 2.02),
                arrowprops={"arrowstyle": "-|>", "color": GRAY, "lw": 1.2},
            )
    ax.text(
        6,
        0.55,
        "Authorization requires structural conformance, package integrity,\nand every mandatory scientific block",
        ha="center",
        va="center",
        color=INK,
        weight="bold",
        fontsize=10.5,
    )
    _save(fig, "standards_workflow")


def claim_evidence_matrix() -> None:
    """Render every direct claim-to-evidence relation in profile v2."""

    profile = load_authorization_profile_v2()
    block_names = [block.name for block in profile.evidence_blocks]
    matrix = np.zeros((len(block_names), len(profile.requirements)), dtype=int)
    block_index = {name: index for index, name in enumerate(block_names)}
    for claim_index, requirement in enumerate(profile.requirements):
        for block_name in requirement.required_blocks:
            matrix[block_index[block_name], claim_index] = 1

    labels = [name.replace("_", " ") for name in block_names]
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    ax.imshow(
        matrix,
        cmap=ListedColormap(["#f7f9fa", BLUE]),
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="none",
    )
    ax.set_xticks(np.arange(len(profile.requirements)), [f"C{i}" for i in range(1, 11)])
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Claim type")
    ax.set_ylabel("Evidence block")
    ax.set_xticks(np.arange(-0.5, len(profile.requirements), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(block_names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="y", labelsize=8.4)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.spines[:].set_visible(False)
    ax.set_title(
        "Profile v2 contains 60 direct claim-to-evidence requirements",
        loc="left",
        pad=30,
        color=INK,
        weight="bold",
    )
    _save(fig, "claim_evidence_matrix_v2")


def integrity_ablation() -> None:
    payload = json.loads(ATTACKS.read_text())
    systems = payload["ablation"]["systems"]
    order = [
        "profile_only",
        "hash_only",
        "without_profile_identity_mismatch",
        "without_artifact_hash_mismatch",
        "without_unknown_provenance_reference",
        "without_provenance_cycle",
        "without_duplicate_independent_artifact",
        "without_overlapping_independent_cohorts",
        "without_contradictory_attestation",
        "without_missing_block_lineage",
        "full_integrity",
    ]
    labels = [
        "Profile only",
        "Hash only",
        "No profile identity",
        "No content hash",
        "No resolved lineage",
        "No cycle check",
        "No duplicate check",
        "No cohort-overlap check",
        "No attestation check",
        "No block lineage",
        "Full gate",
    ]
    values = [systems[name]["false_authorizations"] for name in order]
    colors = [RED, AMBER, *([GRAY] * 8), TEAL]
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, height=0.66)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 385)
    ax.set_xlabel("False authorizations among 360 attacked packages")
    ax.set_title("Integrity controls are non-compensatory")
    ax.grid(axis="x", color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + 5 if value else 4,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            color=INK,
            weight="bold",
        )
    _save(fig, "integrity_ablation_v2")


def scalability() -> None:
    payload = json.loads(ATTACKS.read_text())["scalability"]
    batch = payload["batch_scaling"]
    artifact = payload["artifact_scaling"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9))
    x = [row["packages"] for row in batch]
    throughput = [row["median_packages_per_second"] for row in batch]
    axes[0].plot(x, throughput, marker="o", color=BLUE, linewidth=2)
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, 75000)
    axes[0].set_xlabel("Packages per batch")
    axes[0].set_ylabel("Packages / second")
    axes[0].set_title("Batch throughput")
    axes[0].grid(color=LIGHT)
    x2 = [row["artifacts"] for row in artifact]
    latency = [1000 * row["median_seconds"] for row in artifact]
    axes[1].plot(x2, latency, marker="o", color=TEAL, linewidth=2)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Artifacts in one package")
    axes[1].set_ylabel("Median latency (ms)")
    axes[1].set_title("Integrity-gate latency")
    axes[1].grid(color=LIGHT, which="both")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2.4)
    _save(fig, "scalability_v2")


def prospective_applications() -> None:
    payload = json.loads(DANDI.read_text())
    ach, contrast = payload["applications"]
    aggregate = contrast["aggregate"]
    improvement = (aggregate["baseline_sse"] - aggregate["model_sse"]) / aggregate[
        "baseline_sse"
    ]
    labels = ["Median r", "Bootstrap lower", "Positive mice", "MSE gain"]
    observed = [
        aggregate["median_subject_correlation"],
        aggregate["bootstrap_lower_95"],
        aggregate["positive_subject_fraction"],
        improvement,
    ]
    thresholds = [0.10, 0.0, 0.65, 0.0]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.8, 4.0),
        gridspec_kw={"width_ratios": [0.8, 1.7]},
    )
    axes[0].bar(
        ["Usable", "Required"],
        [ach["usable_paired_subjects"], ach["minimum_subjects"]],
        color=[RED, GRAY],
        width=0.62,
    )
    axes[0].set_ylim(0, 23)
    axes[0].set_ylabel("Subjects")
    axes[0].set_title("DANDI:001176\nnot authorized")
    axes[0].spines[["top", "right"]].set_visible(False)
    for index, value in enumerate([ach["usable_paired_subjects"], ach["minimum_subjects"]]):
        axes[0].text(index, value + 0.6, str(value), ha="center", weight="bold")
    positions = np.arange(len(labels))
    bars = axes[1].bar(positions, observed, color=[BLUE, BLUE, TEAL, TEAL], width=0.62)
    axes[1].scatter(
        positions,
        thresholds,
        color=RED,
        marker="D",
        s=34,
        zorder=3,
        clip_on=False,
    )
    axes[1].set_xticks(positions, labels, rotation=18, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("DANDI:000039\nbounded prediction authorized")
    axes[1].grid(axis="y", color=LIGHT)
    axes[1].spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, observed, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
        )
    fig.tight_layout(w_pad=2.5)
    _save(fig, "prospective_applications_v2")


def main() -> None:
    _style()
    workflow()
    claim_evidence_matrix()
    integrity_ablation()
    scalability()
    prospective_applications()


if __name__ == "__main__":
    main()

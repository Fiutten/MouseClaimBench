"""Audit frozen claim authorization on official CausalBench Perturb-seq data.

The bounded adapter does not compete with causal-discovery algorithms. It asks
whether a claim policy developed on synthetic cases remains selective when
pair-specific intervention evidence is supplied from a different domain. Gene
selection uses K562 strong-perturbation metadata and cross-domain availability,
not RPE1 expression outcomes. Two disjoint cell folds construct evidence and a
third fold supplies reference labels. No synthetic finite-sample certificate is
claimed to transport to either cell line.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse, stats

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.hybrid_development_features import encode_hybrid_features
from mousebrainbench.benchmarks.hybrid_selective_policy import (
    predict_probabilities,
    semantic_admissibility_matrix,
)
from mousebrainbench.knowledge import load_default_profile
from mousebrainbench.validation.evidence_contract import (
    EvidenceBlock,
    EvidenceStatus,
    blocks_by_name,
)
from mousebrainbench.validation.semantic_risk_control import (
    SemanticRiskPolicy,
    authorize_with_policy,
    semantic_false_authorization_metrics,
)

DEFAULT_ROOT = Path("data/external/causalbench")
DEFAULT_SELECTION = Path("configs/benchmarks/causalbench_v3_selection.json")
DEFAULT_SCORE_MODEL = Path("results/hybrid_selective_policy/model.json")
DEFAULT_RISK_POLICY = Path("results/semantic_risk_policy/model.json")
DEFAULT_OUTPUT = Path("results/causalbench_transport/summary.json")
DEFAULT_MARKDOWN = Path("results/causalbench_transport/summary.md")

DATASETS = {
    "weissmann_k562": {
        "filename": "k562.h5ad",
        "expected_bytes": 10_661_879_995,
        "expected_md5": "4f1122ce1c7f13299a68df6459a266d3",
        "role": "calibration_context",
    },
    "weissmann_rpe1": {
        "filename": "rpe1.h5ad",
        "expected_bytes": 8_700_873_216,
        "expected_md5": "6a2a9d0d2bf4ec147f4d1104043b268c",
        "role": "locked_transport_test",
    },
}
SUMMARY_FILENAME = "summary_stats.xlsx"
SUMMARY_SHEET_K562 = "TabB_K562_day6_summary_stat"
MAX_GENES = 200
MIN_CELLS_PER_INTERVENTION = 90
MAX_CELLS_PER_FOLD = 64
FOLD_COUNT = 3
REFERENCE_FOLD = 2
FDR_LEVEL = 0.05
MIN_ABSOLUTE_STANDARDIZED_EFFECT = 0.10
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_SEED = 2_026_080_202
OFFICIAL_REPOSITORY_REVISION = "1a2143cffdc85f835b41ce8d52034be1bf903e71"
FIGSHARE_DOI = "10.25452/figshare.plus.20029387.v1"


def _anndata_module():
    try:
        import anndata
    except ImportError as exc:
        raise RuntimeError(
            "CausalBench transport requires the `semantic-risk-v3` dependencies"
        ) from exc
    return anndata


def _file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}:{value}".encode()).hexdigest()


def deterministic_fold_indices(
    cell_ids: Sequence[str],
    *,
    namespace: str,
    maximum_per_fold: int = MAX_CELLS_PER_FOLD,
) -> tuple[np.ndarray, ...]:
    """Return balanced, disjoint cell folds ordered by a stable cryptographic hash."""

    ordered = sorted(
        enumerate(map(str, cell_ids)),
        key=lambda item: _stable_key(namespace, item[1]),
    )
    folds: list[list[int]] = [[] for _ in range(FOLD_COUNT)]
    for rank, (index, _) in enumerate(ordered):
        fold = rank % FOLD_COUNT
        if len(folds[fold]) < maximum_per_fold:
            folds[fold].append(index)
    return tuple(np.asarray(values, dtype=np.int64) for values in folds)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values with finite-value checks."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("BH correction requires a finite one-dimensional array")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _strong_k562_genes(summary_path: Path) -> set[str]:
    frame = pd.read_excel(summary_path, sheet_name=SUMMARY_SHEET_K562)
    keep = (
        (frame["Number of DEGs (anderson-darling)"] > 50)
        & (frame["percent knockdown"] <= -0.3)
        & (frame["number of cells (filtered)"] > 25)
    )
    # CausalBench filters the same rows by gene symbol. The H5AD intervention
    # and variable indices use Ensembl IDs, stored as the final token here.
    return {
        value.rsplit("_", maxsplit=1)[-1]
        for value in frame.loc[keep, "genetic perturbation"].astype(str)
        if "_" in value
    }


def _metadata(path: Path) -> tuple[set[str], set[str], Mapping[str, int]]:
    anndata = _anndata_module()
    data = anndata.read_h5ad(path, backed="r")
    try:
        if "gene_id" not in data.obs:
            raise ValueError(f"{path} has no CausalBench `gene_id` intervention column")
        interventions = data.obs["gene_id"].astype(str)
        counts = interventions.value_counts().to_dict()
        return set(map(str, data.var_names)), set(interventions), {
            str(gene): int(count) for gene, count in counts.items()
        }
    finally:
        data.file.close()


def prepare_selection(
    *,
    root: Path = DEFAULT_ROOT,
    output: Path = DEFAULT_SELECTION,
) -> Path:
    """Freeze an outcome-independent, bounded cross-domain gene selection."""

    strong = _strong_k562_genes(root / SUMMARY_FILENAME)
    metadata: dict[str, dict[str, Any]] = {}
    eligible = set(strong)
    for dataset, specification in DATASETS.items():
        path = root / str(specification["filename"])
        variables, interventions, counts = _metadata(path)
        eligible &= variables
        eligible &= {
            gene
            for gene in interventions
            if counts.get(gene, 0) >= MIN_CELLS_PER_INTERVENTION
        }
        metadata[dataset] = {
            "variables": len(variables),
            "intervention_labels": len(interventions),
            "eligible_interventions_before_intersection": sum(
                count >= MIN_CELLS_PER_INTERVENTION
                for gene, count in counts.items()
                if gene != "non-targeting"
            ),
        }

    genes = sorted(eligible, key=lambda gene: _stable_key("causalbench-v3", gene))[
        :MAX_GENES
    ]
    if len(genes) < 20:
        raise RuntimeError("fewer than 20 outcome-independent cross-domain genes are eligible")

    source_files = {}
    for dataset, specification in DATASETS.items():
        path = root / str(specification["filename"])
        actual_md5 = _file_hash(path, "md5")
        expected_md5 = str(specification["expected_md5"])
        if actual_md5 != expected_md5 or path.stat().st_size != specification["expected_bytes"]:
            raise ValueError(f"official CausalBench checksum/size mismatch: {path}")
        source_files[dataset] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "md5": actual_md5,
            "expected_md5": expected_md5,
        }

    payload = {
        "protocol": "causalbench_bounded_transport_v3",
        "selection_revision": code_revision(),
        "selection_uses_rpe1_expression_outcomes": False,
        "selection_basis": (
            "official K562 strong-perturbation rule, intervention cell counts, "
            "cross-domain intervention availability, and SHA-256 ordering"
        ),
        "maximum_genes": MAX_GENES,
        "selected_gene_count": len(genes),
        "selected_genes": genes,
        "metadata": metadata,
        "source_files": source_files,
        "summary_stats_sha256": _file_hash(root / SUMMARY_FILENAME, "sha256"),
        "official_repository_revision": OFFICIAL_REPOSITORY_REVISION,
        "figshare_doi": FIGSHARE_DOI,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return output


def _dense(values: Any) -> np.ndarray:
    if sparse.issparse(values):
        return np.asarray(values.toarray(), dtype=float)
    return np.asarray(values, dtype=float)


def _normalized_expression(
    data: Any,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
) -> np.ndarray:
    """Read bounded rows and normalize raw counts in chunks without loading the H5AD."""

    output = np.empty((len(row_indices), len(column_indices)), dtype=np.float64)
    recorded_library_size = (
        data.obs["UMI_count"].to_numpy(dtype=float)
        if "UMI_count" in data.obs
        else None
    )
    for start in range(0, len(row_indices), 512):
        stop = min(start + 512, len(row_indices))
        rows = row_indices[start:stop]
        target_counts = _dense(data[rows, column_indices].X)
        if recorded_library_size is not None:
            library_size = recorded_library_size[rows].copy()
        else:
            full_counts = data[rows, :].X
            library_size = np.asarray(full_counts.sum(axis=1), dtype=float).reshape(-1)
        library_size[library_size <= 0.0] = 1.0
        output[start:stop] = np.log1p(target_counts * (10_000.0 / library_size[:, None]))
    return output


def _domain_fold_matrices(
    path: Path,
    genes: tuple[str, ...],
) -> tuple[dict[str, tuple[np.ndarray, ...]], dict[str, Any]]:
    """Extract only selected intervention cells and variables from a backed H5AD."""

    anndata = _anndata_module()
    data = anndata.read_h5ad(path, backed="r")
    try:
        interventions = data.obs["gene_id"].astype(str).to_numpy()
        cell_ids = np.asarray(data.obs_names.astype(str))
        variable_index = {str(name): index for index, name in enumerate(data.var_names)}
        missing = [gene for gene in genes if gene not in variable_index]
        if missing:
            raise ValueError(f"selected CausalBench genes absent from {path}: {missing[:5]}")

        groups = ("non-targeting", *genes)
        group_rows: dict[str, tuple[np.ndarray, ...]] = {}
        all_rows: list[int] = []
        for group in groups:
            absolute = np.flatnonzero(interventions == group)
            if len(absolute) < MIN_CELLS_PER_INTERVENTION:
                raise ValueError(f"{path}: intervention {group} has only {len(absolute)} cells")
            relative_folds = deterministic_fold_indices(
                cell_ids[absolute], namespace=f"{path.name}:{group}"
            )
            folds = tuple(absolute[relative] for relative in relative_folds)
            if min(map(len, folds)) < 20:
                raise ValueError(f"{path}: intervention {group} has an undersized fold")
            group_rows[group] = folds
            all_rows.extend(int(value) for fold in folds for value in fold)

        ordered_rows = np.asarray(sorted(set(all_rows)), dtype=np.int64)
        row_position = {int(row): index for index, row in enumerate(ordered_rows)}
        matrix = _normalized_expression(
            data,
            ordered_rows,
            np.asarray([variable_index[gene] for gene in genes], dtype=np.int64),
        )
        fold_matrices = {
            group: tuple(
                matrix[np.asarray([row_position[int(row)] for row in fold], dtype=np.int64)]
                for fold in folds
            )
            for group, folds in group_rows.items()
        }
        return fold_matrices, {
            "n_obs": int(data.n_obs),
            "n_vars": int(data.n_vars),
            "selected_cells": len(ordered_rows),
            "normalization": "log1p(raw_count / per_cell_library_size * 10000)",
        }
    finally:
        data.file.close()


def _standardized_effect(intervention: np.ndarray, control: np.ndarray) -> np.ndarray:
    difference = np.mean(intervention, axis=0) - np.mean(control, axis=0)
    pooled_variance = (
        (len(intervention) - 1) * np.var(intervention, axis=0, ddof=1)
        + (len(control) - 1) * np.var(control, axis=0, ddof=1)
    ) / max(len(intervention) + len(control) - 2, 1)
    denominator = np.sqrt(np.maximum(pooled_variance, 1e-12))
    return difference / denominator


def intervention_effect_matrices(
    fold_matrices: Mapping[str, tuple[np.ndarray, ...]],
    genes: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Calculate fold-specific effects and globally FDR-adjusted pair tests."""

    gene_count = len(genes)
    effects = np.zeros((FOLD_COUNT, gene_count, gene_count), dtype=float)
    p_values = np.ones_like(effects)
    controls = fold_matrices["non-targeting"]
    for source_index, source in enumerate(genes):
        for fold in range(FOLD_COUNT):
            intervention = fold_matrices[source][fold]
            control = controls[fold]
            effects[fold, source_index] = _standardized_effect(intervention, control)
            result = stats.mannwhitneyu(
                intervention,
                control,
                alternative="two-sided",
                axis=0,
                method="asymptotic",
            )
            p_values[fold, source_index] = np.asarray(result.pvalue, dtype=float)
            p_values[fold, source_index, source_index] = 1.0
            effects[fold, source_index, source_index] = 0.0

    q_values = np.ones_like(p_values)
    off_diagonal = ~np.eye(gene_count, dtype=bool)
    for fold in range(FOLD_COUNT):
        q_values[fold, off_diagonal] = benjamini_hochberg(
            p_values[fold, off_diagonal]
        )
    return {"effect": effects, "p_value": p_values, "q_value": q_values}


def _block(
    name: str,
    status: EvidenceStatus,
    rule: str,
    observations: Mapping[str, Any] | None = None,
) -> EvidenceBlock:
    return EvidenceBlock.from_mapping(
        name=name,
        status=status,
        source="official CausalBench Perturb-seq bounded adapter",
        rule=rule,
        rationale=f"the pair-specific CausalBench rule returned `{status.value}`",
        observations=dict(observations or {}),
    )


def _pair_feature_and_label(
    *,
    dataset: str,
    source: str,
    target: str,
    source_index: int,
    target_index: int,
    matrices: Mapping[str, np.ndarray],
    sample_size: int,
    claim_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    effect = matrices["effect"][:, source_index, target_index]
    p_value = matrices["p_value"][:, source_index, target_index]
    q_value = matrices["q_value"][:, source_index, target_index]
    evidence_pass = bool(
        np.all(q_value[:2] <= FDR_LEVEL)
        and np.all(np.abs(effect[:2]) >= MIN_ABSOLUTE_STANDARDIZED_EFFECT)
        and np.sign(effect[0]) == np.sign(effect[1])
    )
    reference_pass = bool(
        q_value[REFERENCE_FOLD] <= FDR_LEVEL
        and abs(effect[REFERENCE_FOLD]) >= MIN_ABSOLUTE_STANDARDIZED_EFFECT
    )
    status = EvidenceStatus.PASSED if evidence_pass else EvidenceStatus.FAILED
    unavailable = EvidenceStatus.NOT_APPLICABLE
    observations = {
        "dataset": dataset,
        "source": source,
        "target": target,
        "evidence_effects": effect[:2].tolist(),
        "evidence_p_values": p_value[:2].tolist(),
        "evidence_q_values": q_value[:2].tolist(),
        "reference_effect": float(effect[REFERENCE_FOLD]),
        "reference_p_value": float(p_value[REFERENCE_FOLD]),
        "reference_q_value": float(q_value[REFERENCE_FOLD]),
        "reference_used_as_policy_input": False,
    }
    blocks = blocks_by_name(
        (
            _block(
                "prediction",
                unavailable,
                "no predictive model is evaluated in the Perturb-seq pair adapter",
            ),
            _block(
                "reproducible_compute",
                EvidenceStatus.PASSED,
                "checksum-verified source, frozen genes, and deterministic cell folds",
            ),
            _block(
                "internal_reproduction",
                status,
                "both evidence folds pass BH q <= 0.05, |d| >= 0.10, and sign agreement",
                observations,
            ),
            _block(
                "external_replication",
                unavailable,
                "different cell lines are a transport test, not identical-population replication",
            ),
            _block(
                "topology_specificity",
                unavailable,
                "the adapter does not compare candidate graph topologies",
            ),
            _block(
                "directed_identifiability",
                status,
                "controlled source perturbation changes target expression in two evidence folds",
                observations,
            ),
            _block(
                "structure_function_association",
                unavailable,
                "single-cell expression is outside the mouse structure-function claim",
            ),
            _block(
                "causal_intervention",
                status,
                "controlled source perturbation passes two fold-specific effect tests",
                {
                    **observations,
                    "available": True,
                    "mean_do_plus_minus_do_minus": float(np.mean(np.abs(effect[:2]))),
                    "p_value": float(max(q_value[:2])),
                },
            ),
            _block("whole_brain_coverage", unavailable, "cell lines are not a brain"),
            _block(
                "independent_validation",
                EvidenceStatus.PASSED,
                "CausalBench was excluded from score-model and risk-policy fitting",
            ),
            _block("entity_specificity", unavailable, "no digital-twin entity is defined"),
            _block("operational_compute", unavailable, "no digital-twin runtime is defined"),
        )
    )
    directional = {
        "method": "controlled_genetic_intervention",
        "attempted": True,
        "predicted_direction": "forward" if evidence_pass else "uncertain",
        "status": status.value,
        "p_forward": float(1.0 - max(q_value[:2])) if evidence_pass else 0.0,
        "p_backward": 0.0,
        "signed_margin": float(np.mean(effect[:2])),
        "absolute_margin": float(np.mean(np.abs(effect[:2]))),
    }
    legacy = {
        "forward_residual_dependence": 0.0,
        "reverse_residual_dependence": 0.0,
        "direction_margin": 0.0,
    }
    true_claims = {"computationally_reproducible"}
    if reference_pass:
        true_claims |= {"internally_reproduced", "directed", "causal"}
    feature = encode_hybrid_features(
        blocks,
        legacy_direction=legacy,
        anm=directional,
        sample_size=sample_size,
        noise_scale=0.0,
    )
    label = np.asarray([claim in true_claims for claim in claim_names], dtype=np.uint8)
    return feature, label, observations


def _cluster_bootstrap(
    decisions: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
) -> dict[str, Any]:
    """Quantify stability by resampling source-gene clusters, not pair rows."""

    unique_sources = np.unique(sources)
    groups = {source: np.flatnonzero(sources == source) for source in unique_sources}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    coverage = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    sfar = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique_sources, size=len(unique_sources), replace=True)
        indices = np.concatenate([groups[source] for source in sampled])
        metrics = semantic_false_authorization_metrics(decisions[indices], labels[indices])
        coverage[replicate] = metrics["supported_coverage"]
        sfar[replicate] = metrics["semantic_false_authorization_risk"]
    return {
        "unit": "source_gene_cluster",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "coverage_interval_95": np.quantile(coverage, [0.025, 0.975]).tolist(),
        "sfar_interval_95": np.quantile(sfar, [0.025, 0.975]).tolist(),
        "limitation": (
            "source clustering does not remove all dependence induced by shared targets "
            "and non-targeting control cells"
        ),
    }


def _evaluate_domain(
    *,
    dataset: str,
    path: Path,
    genes: tuple[str, ...],
    score_model: Mapping[str, Any],
    risk_policy: SemanticRiskPolicy,
    variable_claims: tuple[str, ...],
) -> dict[str, Any]:
    fold_matrices, extraction = _domain_fold_matrices(path, genes)
    matrices = intervention_effect_matrices(fold_matrices, genes)
    claim_names = tuple(score_model["claim_names"])
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sources: list[str] = []
    reference_positive_pairs = 0
    for source_index, source in enumerate(genes):
        sample_size = min(len(fold) for fold in fold_matrices[source])
        for target_index, target in enumerate(genes):
            if source == target:
                continue
            feature, label, observations = _pair_feature_and_label(
                dataset=dataset,
                source=source,
                target=target,
                source_index=source_index,
                target_index=target_index,
                matrices=matrices,
                sample_size=sample_size,
                claim_names=claim_names,
            )
            features.append(feature)
            labels.append(label)
            sources.append(source)
            reference_positive_pairs += observations["reference_q_value"] <= FDR_LEVEL

    feature_matrix = np.vstack(features)
    label_matrix = np.vstack(labels).astype(bool)
    source_array = np.asarray(sources)
    probabilities = predict_probabilities(
        score_model["model_sets"]["full"], feature_matrix, claim_names
    )
    requirements = {
        item.claim: item.required_blocks for item in load_default_profile().requirements
    }
    admissible = semantic_admissibility_matrix(
        feature_matrix,
        claim_names=claim_names,
        feature_names=score_model["feature_names"],
        support_vetoes=requirements,
    )
    variable_indices = np.asarray([claim_names.index(claim) for claim in variable_claims])
    decisions = authorize_with_policy(
        risk_policy,
        probabilities[:, variable_indices],
        admissible[:, variable_indices],
    )
    truths = label_matrix[:, variable_indices]
    metrics = semantic_false_authorization_metrics(decisions, truths)
    by_claim = {}
    for index, claim in enumerate(variable_claims):
        by_claim[claim] = semantic_false_authorization_metrics(
            decisions[:, [index]], truths[:, [index]]
        )
    return {
        "dataset": dataset,
        "role": DATASETS[dataset]["role"],
        "extraction": extraction,
        "pair_cases": len(features),
        "reference_positive_pairs": int(reference_positive_pairs),
        **metrics,
        "by_claim": by_claim,
        "semantic_support_violations": int(
            ((decisions == 1) & ~admissible[:, variable_indices]).sum()
        ),
        "source_cluster_bootstrap": _cluster_bootstrap(decisions, truths, source_array),
        "guarantee_transport_valid": False,
    }


def run(
    *,
    root: Path = DEFAULT_ROOT,
    selection_path: Path = DEFAULT_SELECTION,
    score_model_path: Path = DEFAULT_SCORE_MODEL,
    risk_policy_path: Path = DEFAULT_RISK_POLICY,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    """Run K562 context and the locked RPE1 transport audit without refitting."""

    selection = json.loads(selection_path.read_text())
    if selection.get("selection_uses_rpe1_expression_outcomes") is not False:
        raise ValueError("CausalBench selection must explicitly exclude RPE1 outcomes")
    genes = tuple(map(str, selection["selected_genes"]))
    if not 20 <= len(genes) <= MAX_GENES:
        raise ValueError("CausalBench frozen selection has an invalid gene count")

    score_model = json.loads(score_model_path.read_text())
    frozen_policy = json.loads(risk_policy_path.read_text())
    risk_policy = SemanticRiskPolicy.from_dict(frozen_policy["semantic_policy"])
    variable_claims = tuple(frozen_policy["variable_claims"])
    domains = [
        _evaluate_domain(
            dataset=dataset,
            path=root / str(specification["filename"]),
            genes=genes,
            score_model=score_model,
            risk_policy=risk_policy,
            variable_claims=variable_claims,
        )
        for dataset, specification in DATASETS.items()
    ]
    locked = next(row for row in domains if row["role"] == "locked_transport_test")
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "causalbench_interventional_transport_audit_v3",
        "official_repository_revision": OFFICIAL_REPOSITORY_REVISION,
        "figshare_doi": FIGSHARE_DOI,
        "anndata_version": importlib.metadata.version("anndata"),
        "selection_path": str(selection_path),
        "selection_sha256": _file_hash(selection_path, "sha256"),
        "selected_gene_count": len(genes),
        "pair_cases_per_domain": len(genes) * (len(genes) - 1),
        "evidence_folds": [0, 1],
        "reference_fold": REFERENCE_FOLD,
        "reference_used_as_policy_input": False,
        "multiple_testing": "Benjamini-Hochberg within domain and cell fold",
        "effect_threshold": MIN_ABSOLUTE_STANDARDIZED_EFFECT,
        "score_model_refitted": False,
        "risk_policy_recalibrated": False,
        "synthetic_guarantee_transported": False,
        "variable_claims": list(variable_claims),
        "domains": domains,
        "locked_test_zero_semantic_violations": locked["semantic_support_violations"] == 0,
        "decision": (
            "causalbench_transport_completed_without_semantic_violation"
            if locked["semantic_support_violations"] == 0
            else "causalbench_transport_semantic_failure"
        ),
        "limits": [
            "Cross-cell-line results are transport evidence, not a transported LTT guarantee.",
            "Perturbation effects can reflect indirect paths, off-target effects, or cell-state shifts.",
            "The adapter evaluates a bounded gene subset and is not a causal-discovery leaderboard.",
            "Source-gene bootstrap intervals retain dependence through shared targets and controls.",
            "A failed authorization is abstention and does not establish absence of an effect.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(payload, markdown)
    return output


def _write_markdown(payload: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# CausalBench interventional transport audit",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Selected genes: `{payload['selected_gene_count']}`",
        f"- Directed pairs per domain: `{payload['pair_cases_per_domain']}`",
        "- Synthetic finite-sample guarantee transported: `false`",
        "",
        "| Domain | Role | Authorizations | Coverage | SFAR | Semantic violations |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["domains"]:
        lines.append(
            f"| `{row['dataset']}` | `{row['role']}` | {row['authorizations']} | "
            f"{row['supported_coverage']:.4f} | "
            f"{row['semantic_false_authorization_risk']:.4f} | "
            f"{row['semantic_support_violations']} |"
        )
    lines.extend(("", "## Limits", ""))
    lines.extend(f"- {limit}" for limit in payload["limits"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--score-model", type=Path, default=DEFAULT_SCORE_MODEL)
    parser.add_argument("--risk-policy", type=Path, default=DEFAULT_RISK_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--prepare-selection",
        action="store_true",
        help="freeze metadata-only gene selection and exit without reading expression outcomes",
    )
    args = parser.parse_args()
    if args.prepare_selection:
        path = prepare_selection(root=args.root, output=args.selection)
    else:
        path = run(
            root=args.root,
            selection_path=args.selection,
            score_model_path=args.score_model,
            risk_policy_path=args.risk_policy,
            output=args.output,
            markdown=args.markdown,
        )
    print(json.dumps({"output": str(path.resolve())}))


if __name__ == "__main__":
    main()

"""Descriptive transport of the fixed topology gate to CausalRivers.

CausalRivers contains many dependent station pairs but only three geographical
clusters. This module therefore reports block and matched-shift summaries. It
does not compute a pair-level confidence interval or an external risk
certificate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.causal_chambers_transport import _pair_record
from mousebrainbench.benchmarks.hybrid_selective_policy import (
    predict_probabilities,
    semantic_admissibility_matrix,
)
from mousebrainbench.knowledge import load_default_profile
from mousebrainbench.validation.shift_diagnostics import diagnose_shift

DEFAULT_PROTOCOL = Path("configs/benchmarks/causalrivers_v5_transport.yaml")
DEFAULT_ROOT = Path("data/external/causalrivers_v5/product")
DEFAULT_SCORE_MODEL = Path("results/hybrid_selective_policy/model.json")
DEFAULT_RISK_POLICY = Path("results/semantic_risk_policy/model.json")
DEFAULT_OUTPUT = Path("results/causalrivers_v5_transport/summary.json")
DEFAULT_MARKDOWN = Path("results/causalrivers_v5_transport/summary.md")


@dataclass(frozen=True)
class BlockData:
    """One real-data block after fixed selection and deterministic exclusion."""

    block_id: str
    role: str
    records: tuple[dict[str, Any], ...]
    scores: np.ndarray
    labels: np.ndarray
    admissible: np.ndarray
    features: np.ndarray
    selected_pairs: tuple[tuple[str, str, bool], ...]
    sampled_rows: int
    source_rows: int
    loaded_columns: int
    exclusions: dict[str, int]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_sources(protocol: dict[str, Any], root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected in protocol["source"]["file_sha256"].items():
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"missing frozen CausalRivers file: {path}")
        digest = _sha256(path)
        if digest != str(expected):
            raise RuntimeError(f"CausalRivers hash mismatch for {name}: {digest}")
        observed[name] = f"sha256:{digest}"
    return observed


def _load_graph(path: Path) -> nx.DiGraph:
    """Load a hash-verified official graph pickle from the trusted release."""

    with path.open("rb") as stream:
        # Pickle is accepted only after the official release hash is verified.
        graph = pickle.load(stream)
    if not isinstance(graph, nx.DiGraph):
        raise TypeError(f"expected a directed NetworkX graph in {path}")
    return graph


def _csv_columns(path: Path) -> tuple[str, ...]:
    names = tuple(str(name) for name in pd.read_csv(path, nrows=0).columns)
    return tuple(name for name in names if name != "datetime")


def select_graph_pairs(
    graph: nx.DiGraph,
    columns: tuple[str, ...],
    *,
    namespace: str,
    direct_count: int,
    control_count: int,
) -> tuple[tuple[str, str, bool], ...]:
    """Select balanced edges and nonedges using identifiers, never values."""

    available = set(columns)
    edges = {
        (str(source), str(target))
        for source, target in graph.edges
        if str(source) in available and str(target) in available
    }
    direct = sorted(
        edges,
        key=lambda pair: hashlib.sha256(
            f"{namespace}:edge:{pair[0]}:{pair[1]}".encode()
        ).hexdigest(),
    )[:direct_count]
    forbidden = edges | {(target, source) for source, target in edges}
    controls = sorted(
        (
            (source, target)
            for source in columns
            for target in columns
            if source != target and (source, target) not in forbidden
        ),
        key=lambda pair: hashlib.sha256(
            f"{namespace}:nonedge:{pair[0]}:{pair[1]}".encode()
        ).hexdigest(),
    )[:control_count]
    if len(direct) != direct_count or len(controls) != control_count:
        raise RuntimeError(
            f"insufficient fixed pairs in {namespace}: {len(direct)} edges, "
            f"{len(controls)} controls"
        )
    return tuple(
        [(source, target, True) for source, target in direct]
        + [(source, target, False) for source, target in controls]
    )


def _adapter_controls(
    eligible: tuple[str, ...],
    *,
    dataset: str,
    experiment: str,
    source: str,
    target: str,
) -> tuple[str, ...]:
    candidates = [name for name in eligible if name not in {source, target}]
    return tuple(
        sorted(
            candidates,
            key=lambda name: hashlib.sha256(
                f"{dataset}:{experiment}:{source}:{target}:{name}".encode()
            ).hexdigest(),
        )[:3]
    )


def _read_selected_frame(
    path: Path,
    *,
    eligible: tuple[str, ...],
    pairs: tuple[tuple[str, str, bool], ...],
    dataset: str,
    experiment: str,
    sampled_rows: int,
) -> tuple[pd.DataFrame, int, int]:
    required = {"datetime"}
    for source, target, _direct in pairs:
        required.update((source, target))
        required.update(
            _adapter_controls(
                eligible,
                dataset=dataset,
                experiment=experiment,
                source=source,
                target=target,
            )
        )
    frame = pd.read_csv(path, usecols=sorted(required))
    source_rows = len(frame)
    if source_rows > sampled_rows:
        indices = np.linspace(0, source_rows - 1, sampled_rows, dtype=int)
        frame = frame.iloc[indices].reset_index(drop=True)
    return frame, source_rows, len(required)


def _build_block(
    block: dict[str, Any],
    *,
    root: Path,
    pairs: tuple[tuple[str, str, bool], ...],
    score_model: dict[str, Any],
    claim: str,
    sampled_rows: int,
) -> BlockData:
    block_id = str(block["id"])
    dataset = "causalrivers-v5"
    csv_path = root / str(block["time_series"])
    eligible = _csv_columns(csv_path)
    frame, source_rows, loaded_columns = _read_selected_frame(
        csv_path,
        eligible=eligible,
        pairs=pairs,
        dataset=dataset,
        experiment=block_id,
        sampled_rows=sampled_rows,
    )
    sampled_count = len(frame)
    claim_names = tuple(str(value) for value in score_model["claim_names"])
    records: list[dict[str, Any]] = []
    exclusions = {"pair_adapter_failed": 0}
    for source, target, direct_edge in pairs:
        record = _pair_record(
            frame,
            dataset=dataset,
            chamber="causalrivers",
            experiment=block_id,
            source=source,
            target=target,
            direct_edge=direct_edge,
            eligible=eligible,
            claim_names=claim_names,
        )
        if record is None:
            exclusions["pair_adapter_failed"] += 1
            continue
        records.append(
            {
                **record,
                "block_id": block_id,
                "pair_id": f"{source}->{target}",
            }
        )
    del frame
    gc.collect()
    if not records:
        raise RuntimeError(f"no usable CausalRivers pairs in {block_id}")
    features = np.vstack([row["features"] for row in records])
    all_labels = np.vstack([row["labels"] for row in records]).astype(bool)
    probabilities = predict_probabilities(score_model["model_sets"]["full"], features, claim_names)
    requirements = {
        item.claim: item.required_blocks for item in load_default_profile().requirements
    }
    all_admissible = semantic_admissibility_matrix(
        features,
        claim_names=claim_names,
        feature_names=score_model["feature_names"],
        support_vetoes=requirements,
    )
    index = claim_names.index(claim)
    return BlockData(
        block_id=block_id,
        role=str(block["role"]),
        records=tuple(records),
        scores=probabilities[:, index],
        labels=all_labels[:, index],
        admissible=all_admissible[:, index],
        features=features,
        selected_pairs=pairs,
        sampled_rows=sampled_count,
        source_rows=source_rows,
        loaded_columns=loaded_columns,
        exclusions=exclusions,
    )


def _decision_metrics(
    decisions: np.ndarray,
    data: BlockData,
) -> dict[str, Any]:
    chosen = np.asarray(decisions, dtype=bool)
    false = chosen & ~data.labels
    positives = data.labels & data.admissible
    return {
        "usable_pairs": len(data.records),
        "authorizations": int(chosen.sum()),
        "false_authorizations": int(false.sum()),
        "authorization_coverage": float(chosen.mean()),
        "false_authorization_fraction": float(false.sum() / chosen.sum())
        if chosen.any()
        else 0.0,
        "eligible_positive_pairs": int(positives.sum()),
        "recovered_positive_pairs": int((chosen & data.labels).sum()),
        "empirical_positive_recovery": float((chosen & data.labels).sum() / positives.sum())
        if positives.any()
        else 0.0,
        "semantic_violations": int((chosen & ~data.admissible).sum()),
        "inferential_status": "descriptive_only_shared_network_and_stations",
    }


def _comparators(
    data: BlockData,
    *,
    fixed_threshold: float,
    frozen_v3_threshold: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    decisions = {
        "abstain_all": np.zeros_like(data.admissible, dtype=bool),
        "fixed_probability_0_5": data.scores >= 0.5,
        "evidence_contract_only": data.admissible,
        "frozen_v3_topology_threshold": data.admissible
        & (data.scores >= frozen_v3_threshold),
        "v5_1_fixed_hierarchical_threshold": data.admissible
        & (data.scores >= fixed_threshold),
    }
    return (
        {name: _decision_metrics(value, data) for name, value in decisions.items()},
        decisions,
    )


def matched_transition(
    historical: BlockData,
    shifted: BlockData,
    historical_decisions: np.ndarray,
    shifted_decisions: np.ndarray,
) -> dict[str, Any]:
    """Describe decision changes for identical pairs across two regimes."""

    historical_map = {
        row["pair_id"]: (bool(decision), bool(label))
        for row, decision, label in zip(
            historical.records,
            historical_decisions,
            historical.labels,
            strict=True,
        )
    }
    shifted_map = {
        row["pair_id"]: (bool(decision), bool(label))
        for row, decision, label in zip(
            shifted.records,
            shifted_decisions,
            shifted.labels,
            strict=True,
        )
    }
    shared = sorted(set(historical_map) & set(shifted_map))
    transitions = {"off_to_off": 0, "off_to_on": 0, "on_to_off": 0, "on_to_on": 0}
    for pair_id in shared:
        before = historical_map[pair_id][0]
        after = shifted_map[pair_id][0]
        transitions[f"{'on' if before else 'off'}_to_{'on' if after else 'off'}"] += 1
    return {
        "shared_pairs": len(shared),
        "historical_only_pairs": len(set(historical_map) - set(shifted_map)),
        "shift_only_pairs": len(set(shifted_map) - set(historical_map)),
        "transitions": transitions,
        "inferential_status": "descriptive_paired_shared_station_dependence",
    }


def _frozen_v3_threshold(risk_policy: dict[str, Any], claim: str) -> float:
    for row in risk_policy["semantic_policy"]["certificates"]:
        if row["claim"] == claim:
            return float(row["threshold"])
    raise KeyError(claim)


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    root: Path = DEFAULT_ROOT,
    score_model_path: Path = DEFAULT_SCORE_MODEL,
    risk_policy_path: Path = DEFAULT_RISK_POLICY,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    """Evaluate frozen topology claims on historical and flood CausalRivers blocks."""

    started = time.perf_counter()
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol["status"] != "frozen_after_source_metadata_audit_before_transport_outcomes":
        raise ValueError("CausalRivers transport protocol is not frozen")
    hashes = _verify_sources(protocol, root)
    score_model = json.loads(score_model_path.read_text())
    risk_policy = json.loads(risk_policy_path.read_text())
    claim = str(protocol["frozen_policy"]["claim"])
    fixed_threshold = float(protocol["frozen_policy"]["threshold"])
    v3_threshold = _frozen_v3_threshold(risk_policy, claim)
    sampling = protocol["sampling"]
    count_direct = int(sampling["direct_pairs_per_region"])
    count_control = int(sampling["nonedge_controls_per_region"])

    graph_paths = {
        block["id"]: root / str(block["graph"]) for block in protocol["blocks"]
    }
    graphs = {block_id: _load_graph(path) for block_id, path in graph_paths.items()}
    blocks_by_id = {str(block["id"]): block for block in protocol["blocks"]}
    pairs_by_block: dict[str, tuple[tuple[str, str, bool], ...]] = {}
    for block_id in ("bavaria_historical", "east_germany_historical"):
        block = blocks_by_id[block_id]
        columns = _csv_columns(root / str(block["time_series"]))
        pairs_by_block[block_id] = select_graph_pairs(
            graphs[block_id],
            columns,
            namespace=f"{sampling['namespace']}:{block_id}",
            direct_count=count_direct,
            control_count=count_control,
        )
    flood_columns = set(
        _csv_columns(root / str(blocks_by_id["elbe_flood_2024"]["time_series"]))
    )
    historical_columns = set(
        _csv_columns(root / str(blocks_by_id["elbe_historical_matched"]["time_series"]))
    )
    matched_columns = tuple(sorted(flood_columns & historical_columns))
    matched_pairs = select_graph_pairs(
        graphs["elbe_flood_2024"],
        matched_columns,
        namespace=f"{sampling['namespace']}:elbe_matched",
        direct_count=count_direct,
        control_count=count_control,
    )
    pairs_by_block["elbe_historical_matched"] = matched_pairs
    pairs_by_block["elbe_flood_2024"] = matched_pairs

    block_data: dict[str, BlockData] = {}
    comparator_rows: dict[str, dict[str, Any]] = {}
    decision_rows: dict[str, dict[str, np.ndarray]] = {}
    for block in protocol["blocks"]:
        block_id = str(block["id"])
        data = _build_block(
            block,
            root=root,
            pairs=pairs_by_block[block_id],
            score_model=score_model,
            claim=claim,
            sampled_rows=int(sampling["historical_uniform_rows"]),
        )
        block_data[block_id] = data
        comparator_rows[block_id], decision_rows[block_id] = _comparators(
            data,
            fixed_threshold=fixed_threshold,
            frozen_v3_threshold=v3_threshold,
        )

    policy_name = "v5_1_fixed_hierarchical_threshold"
    historical = block_data["elbe_historical_matched"]
    flood = block_data["elbe_flood_2024"]
    transition = matched_transition(
        historical,
        flood,
        decision_rows["elbe_historical_matched"][policy_name],
        decision_rows["elbe_flood_2024"][policy_name],
    )
    shift = diagnose_shift(
        historical.features,
        flood.features,
        feature_names=tuple(score_model["feature_names"]),
    )
    elapsed = time.perf_counter() - started
    result = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "causalrivers_v5_fixed_topology_gate_transport",
        "protocol": str(protocol_path),
        "protocol_status": protocol["status"],
        "source_hashes": hashes,
        "claim": claim,
        "fixed_threshold": fixed_threshold,
        "threshold_refitted": False,
        "independent_top_level_clusters": int(
            protocol["inference"]["independent_top_level_clusters"]
        ),
        "exact_external_certificate_allowed": False,
        "pair_level_confidence_intervals_allowed": False,
        "blocks": {
            block_id: {
                "role": data.role,
                "graph_nodes": len(graphs[block_id]),
                "graph_edges": graphs[block_id].number_of_edges(),
                "selected_pairs": len(data.selected_pairs),
                "usable_pairs": len(data.records),
                "source_rows": data.source_rows,
                "sampled_rows": data.sampled_rows,
                "loaded_columns": data.loaded_columns,
                "exclusions": data.exclusions,
                "comparators": comparator_rows[block_id],
            }
            for block_id, data in block_data.items()
        },
        "matched_flood_shift": {
            "policy": policy_name,
            "transition": transition,
            "feature_shift": shift,
            "historical_metrics": comparator_rows["elbe_historical_matched"][policy_name],
            "flood_metrics": comparator_rows["elbe_flood_2024"][policy_name],
        },
        "efficiency": {
            "wall_time_seconds": elapsed,
            "peak_rss_megabytes": _peak_rss_mb(),
            "source_bytes": sum((root / name).stat().st_size for name in hashes),
        },
        "decision": "descriptive_real_transport_completed_no_external_certificate",
        "interpretation": (
            "CausalRivers tests real-data transport and a matched flood shift. Its "
            "three dependent geographical clusters cannot validate the exact TimeGraph risk bound."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    _write_markdown(result, markdown)
    return output


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# CausalRivers v5 real transport",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Claim: `{payload['claim']}`",
        f"- Independent geographical clusters: `{payload['independent_top_level_clusters']}`",
        "- Exact external certificate: `not allowed`",
        "",
        "| Block | Usable pairs | Authorizations | False | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for block_id, block in payload["blocks"].items():
        row = block["comparators"]["v5_1_fixed_hierarchical_threshold"]
        lines.append(
            f"| `{block_id}` | {block['usable_pairs']} | {row['authorizations']} | "
            f"{row['false_authorizations']} | {row['authorization_coverage']:.4f} |"
        )
    transition = payload["matched_flood_shift"]["transition"]
    lines.extend(
        (
            "",
            "## Matched flood shift",
            "",
            f"- Shared pairs: `{transition['shared_pairs']}`",
            f"- Transitions: `{json.dumps(transition['transitions'], sort_keys=True)}`",
            f"- Shift warning: `{str(payload['matched_flood_shift']['feature_shift']['warning']).lower()}`",
            "",
            payload["interpretation"],
            "",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--score-model", type=Path, default=DEFAULT_SCORE_MODEL)
    parser.add_argument("--risk-policy", type=Path, default=DEFAULT_RISK_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(
        run(
            protocol_path=args.protocol,
            root=args.root,
            score_model_path=args.score_model,
            risk_policy_path=args.risk_policy,
            output=args.output,
            markdown=args.markdown,
        ).resolve()
    )


if __name__ == "__main__":
    main()

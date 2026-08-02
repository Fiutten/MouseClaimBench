"""Audit completed external ratings for the MouseClaimBench knowledge profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.validation.content_validity import dimension_summary

DEFAULT_PROTOCOL = Path("configs/validation/knowledge_profile_external_review_v1.yaml")
DEFAULT_PACKET = Path("docs/knowledge_profile_external_review_v1")
DEFAULT_OUTPUT = Path("results/knowledge_profile_external_validation_v5/summary.json")
DEFAULT_MARKDOWN = Path("results/knowledge_profile_external_validation_v5/summary.md")


def _boolean(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )


def _pending(reason: str, *, items: int, raters: int) -> dict[str, Any]:
    return {
        "profile_content_validated": False,
        "decision": "external_content_validation_pending",
        "reason": reason,
        "items": items,
        "completed_independent_raters": raters,
    }


def evaluate(
    protocol: dict[str, Any],
    items: pd.DataFrame,
    raters: pd.DataFrame,
    ratings: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate panel eligibility, completeness, CVI, agreement, and vetoes."""

    if raters.empty or ratings.empty:
        return _pending("no_external_ratings_received", items=len(items), raters=0)
    required_rater_columns = {
        "rater_id",
        "independent_of_authors",
        "conflict_disclosed",
        "years_relevant_experience",
        "completed",
        *protocol["panel"]["expertise_strata"],
    }
    required_rating_columns = {
        "rater_id",
        "item_id",
        *protocol["rating"]["dimensions"],
        "item_decision",
        "critical_veto",
        "comment",
    }
    if not required_rater_columns.issubset(raters.columns):
        return _pending("rater_metadata_columns_incomplete", items=len(items), raters=0)
    if not required_rating_columns.issubset(ratings.columns):
        return _pending("rating_columns_incomplete", items=len(items), raters=0)

    eligible = raters.copy()
    eligible["independent"] = _boolean(eligible["independent_of_authors"])
    eligible["conflict_ok"] = _boolean(eligible["conflict_disclosed"])
    eligible["is_complete"] = _boolean(eligible["completed"])
    eligible = eligible[
        eligible["independent"].eq(True)
        & eligible["conflict_ok"].eq(True)
        & eligible["is_complete"].eq(True)
        & (
            pd.to_numeric(eligible["years_relevant_experience"], errors="coerce")
            >= int(protocol["panel"]["minimum_years_relevant_experience"])
        )
    ]
    rater_ids = tuple(sorted(eligible["rater_id"].astype(str).unique()))
    minimum = int(protocol["panel"]["minimum_completed_independent_experts"])
    if len(rater_ids) < minimum:
        return _pending(
            "insufficient_eligible_independent_raters",
            items=len(items),
            raters=len(rater_ids),
        )
    expertise_counts = {
        name: int(_boolean(eligible[name]).eq(True).sum())
        for name in protocol["panel"]["expertise_strata"]
    }
    if min(expertise_counts.values()) < int(protocol["panel"]["minimum_per_expertise_stratum"]):
        return _pending("expertise_strata_incomplete", items=len(items), raters=len(rater_ids))

    selected = ratings[ratings["rater_id"].astype(str).isin(rater_ids)].copy()
    selected["rater_id"] = selected["rater_id"].astype(str)
    selected["item_id"] = selected["item_id"].astype(str)
    expected_items = tuple(items["item_id"].astype(str))
    counts = selected.groupby("rater_id")["item_id"].nunique()
    if any(counts.get(rater_id, 0) != len(expected_items) for rater_id in rater_ids):
        return _pending("incomplete_item_response_by_rater", items=len(items), raters=len(rater_ids))
    if selected.duplicated(["rater_id", "item_id"]).any():
        return _pending("duplicate_item_response", items=len(items), raters=len(rater_ids))
    if set(selected["item_id"]) != set(expected_items):
        return _pending("rating_item_set_mismatch", items=len(items), raters=len(rater_ids))

    dimension_rows: dict[str, Any] = {}
    item_threshold = float(protocol["acceptance"]["minimum_item_cvi"])
    scale_threshold = float(protocol["acceptance"]["minimum_scale_average_cvi"])
    kappa_threshold = float(protocol["acceptance"]["minimum_binary_fleiss_kappa"])
    for dimension in protocol["rating"]["dimensions"]:
        pivot = selected.pivot(index="item_id", columns="rater_id", values=dimension)
        pivot = pivot.loc[list(expected_items), list(rater_ids)].apply(
            pd.to_numeric, errors="coerce"
        )
        if pivot.isna().any().any():
            return _pending(
                f"non_numeric_or_missing_{dimension}_rating",
                items=len(items),
                raters=len(rater_ids),
            )
        summary = dimension_summary(pivot.to_numpy(dtype=int), expected_items)
        summary["all_item_cvi_pass"] = all(
            row["item_cvi"] >= item_threshold for row in summary["items"].values()
        )
        summary["scale_cvi_pass"] = summary["scale_average_cvi"] >= scale_threshold
        summary["agreement_pass"] = summary["binary_fleiss_kappa"] >= kappa_threshold
        dimension_rows[str(dimension)] = summary

    veto = _boolean(selected["critical_veto"])
    unresolved_vetoes = int(veto.eq(True).sum())
    veto_comments_complete = bool(
        selected.loc[veto.eq(True), "comment"].astype(str).str.strip().ne("").all()
    )
    content_pass = all(
        row["all_item_cvi_pass"] and row["scale_cvi_pass"] and row["agreement_pass"]
        for row in dimension_rows.values()
    )
    validated = bool(content_pass and unresolved_vetoes == 0 and veto_comments_complete)
    return {
        "profile_content_validated": validated,
        "decision": (
            "external_content_validation_passed"
            if validated
            else "external_panel_complete_profile_revision_required"
        ),
        "items": len(items),
        "completed_independent_raters": len(rater_ids),
        "expertise_counts": expertise_counts,
        "dimensions": dimension_rows,
        "unresolved_critical_vetoes": unresolved_vetoes,
        "critical_veto_comments_complete": veto_comments_complete,
    }


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    packet: Path = DEFAULT_PACKET,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    protocol = yaml.safe_load(protocol_path.read_text())
    items = pd.read_csv(packet / "review_items.csv")
    raters = pd.read_csv(packet / "raters.csv")
    ratings = pd.read_csv(packet / "ratings.csv")
    assessment = evaluate(protocol, items, raters, ratings)
    result = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "knowledge_profile_external_content_validation_v5",
        "protocol": str(protocol_path),
        "packet": str(packet),
        **assessment,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Knowledge-profile external validation",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Content validated: `{str(result['profile_content_validated']).lower()}`",
        f"- Items: `{result['items']}`",
        f"- Completed independent raters: `{result['completed_independent_raters']}`",
        "",
    ]
    if "reason" in result:
        lines.append(f"- Pending reason: `{result['reason']}`")
        lines.append("")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(run(protocol_path=args.protocol, packet=args.packet, output=args.output, markdown=args.markdown).resolve())


if __name__ == "__main__":
    main()

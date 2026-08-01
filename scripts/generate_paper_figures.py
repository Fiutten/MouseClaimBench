"""Generate deterministic manuscript figures without external plotting state.

The paper is compiled in Overleaf and the generated assets are versioned in the
repository.  For that reason this script avoids Matplotlib font-cache side
effects and writes plain PNG files with Pillow.  The figures are intentionally
schematic: they summarize the decision-support workflow and the frozen
experimental evidence without introducing new numerical results.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIRS = (ROOT / "figures", ROOT / "paper" / "figures")

INK = "#111827"
MUTED = "#475569"
LINE = "#334155"
BLUE = "#dcebf7"
GREEN = "#def0e5"
AMBER = "#f5ead0"
RED = "#f6dddd"
GREY = "#f3f4f6"
WHITE = "#ffffff"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a stable system font, falling back to Pillow's default bitmap font."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


HEAD = _font(24, bold=True)
BODY = _font(21)
SMALL = _font(18)
SMALL_BOLD = _font(18, bold=True)


def _wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *,
             font: ImageFont.ImageFont, fill: str = INK, width: int = 32,
             line_gap: int = 6, align: str = "center") -> None:
    """Draw wrapped text with predictable spacing."""
    lines: list[str] = []
    for part in text.split("\n"):
        lines.extend(textwrap.wrap(part, width=width) or [""])
    x, y = xy
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        if align == "center":
            dx = -((bbox[2] - bbox[0]) // 2)
        elif align == "right":
            dx = -(bbox[2] - bbox[0])
        else:
            dx = 0
        draw.text((x + dx, y), line, font=font, fill=fill)
        y += (bbox[3] - bbox[1]) + line_gap


def _box(draw: ImageDraw.ImageDraw, xyxy: tuple[int, int, int, int], title: str,
         body: str, *, fill: str = GREY, outline: str = LINE) -> None:
    """Draw a rounded block used by the workflow and dashboard figures."""
    draw.rounded_rectangle(xyxy, radius=18, fill=fill, outline=outline, width=3)
    x0, y0, x1, y1 = xyxy
    cx = (x0 + x1) // 2
    _wrapped(draw, (cx, y0 + 28), title, font=HEAD, width=22)
    _wrapped(draw, (cx, y0 + 88), body, font=BODY, fill=MUTED, width=26)


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    """Draw a simple arrow that remains legible after scaling in the manuscript."""
    draw.line((start, end), fill=LINE, width=5)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex >= sx else -1
        points = [(ex, ey), (ex - direction * 20, ey - 12), (ex - direction * 20, ey + 12)]
    else:
        direction = 1 if ey >= sy else -1
        points = [(ex, ey), (ex - 12, ey - direction * 20), (ex + 12, ey - direction * 20)]
    draw.polygon(points, fill=LINE)


def build_workflow() -> Image.Image:
    """Create the ClaimBench workflow figure."""
    img = Image.new("RGB", (1800, 760), WHITE)
    draw = ImageDraw.Draw(img)

    boxes = [
        (45, 25, 325, 275, "Source package", "Manuscript\nresults\nprovenance", BLUE),
        (395, 25, 675, 275, "Claim discovery", "Rules or language models\npropose candidates\nwithout authority", GREY),
        (745, 25, 1025, 275, "Domain contract", "Required blocks\ndomain rules\nadmissible wording", GREEN),
        (1095, 25, 1375, 275, "Artifact predicates", "Original-scale values\ncontrols\nsource revision", BLUE),
        (1445, 25, 1725, 275, "Veto gate", "Non-compensatory:\nno block can offset\na failed requirement", AMBER),
    ]
    for x0, y0, x1, y1, title, body, fill in boxes:
        _box(draw, (x0, y0, x1, y1), title, body, fill=fill)
    for start, end in [
        ((325, 150), (395, 150)),
        ((675, 150), (745, 150)),
        ((1025, 150), (1095, 150)),
        ((1375, 150), (1445, 150)),
    ]:
        _arrow(draw, start, end)

    draw.line((1585, 275, 1585, 330), fill=LINE, width=5)
    draw.line((175, 330, 1585, 330), fill=LINE, width=5)
    outcomes = [
        (45, 365, 325, 700, "Supported", "All required blocks pass.", GREEN, "#26734d"),
        (395, 365, 675, 700, "Blocked", "At least one executed block fails.", RED, "#a33a32"),
        (745, 365, 1025, 700, "Uncertain", "A required observation is missing.", AMBER, "#a06b18"),
        (1095, 365, 1375, 700, "Out of scope", "The protocol did not target the block.", GREY, LINE),
        (1445, 365, 1725, 700, "External review", "A non-automatable judgement remains.", BLUE, "#356d95"),
    ]
    for x0, y0, x1, y1, title, body, fill, outline in outcomes:
        draw.line(((x0 + x1) // 2, 330, (x0 + x1) // 2, 365), fill=LINE, width=4)
        _box(draw, (x0, y0, x1, y1), title, body, fill=fill, outline=outline)
    return img


def build_real_case_matrix() -> Image.Image:
    """Create a status matrix for the four artifact-grounded mouse cases."""

    payload = json.loads(
        (ROOT / "results" / "real_case_claim_matrix" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    cases = payload["cases"]
    decisions = {
        (row["case"], row["claim"]): row["status"]
        for row in payload["claim_decisions"]
    }
    claims = [
        ("predictive", "Predictive"),
        ("computationally_reproducible", "Compute\nreproducible"),
        ("internally_reproduced", "Internal\nreproduction"),
        ("externally_replicated", "External\nreplication"),
        ("topology_specific", "Topology\nspecific"),
        ("directed", "Directed"),
        ("structure_function", "Structure-\nfunction"),
        ("mechanistic", "Mechanistic"),
        ("causal", "Causal"),
        ("digital_twin", "Digital\ntwin"),
    ]
    labels = {
        "allen_vbn_identifiability_negative": "Allen VBN",
        "sensorium_static_predictive_topographic": "Static Sensorium",
        "dynamic_sensorium_temporal_prediction": "Dynamic Sensorium",
        "microns_local_structure_function": "MICRONS local",
    }
    colors = {
        "supported": "#4f9a70",
        "blocked": "#c85b58",
        "uncertain": "#d5a340",
        "out_of_scope": "#b7bec7",
        "needs_external_review": "#5790b5",
    }
    img = Image.new("RGB", (2200, 920), WHITE)
    draw = ImageDraw.Draw(img)
    left = 360
    top = 180
    cell_w = 170
    cell_h = 125
    for index, (_claim, label) in enumerate(claims):
        _wrapped(
            draw,
            (left + index * cell_w + cell_w // 2, 55),
            label,
            font=SMALL_BOLD,
            width=14,
        )
    for row_index, case in enumerate(cases):
        y = top + row_index * cell_h
        draw.text(
            (left - 25, y + cell_h // 2),
            labels[case["case"]],
            font=BODY,
            fill=INK,
            anchor="rm",
        )
        for column_index, (claim, _label) in enumerate(claims):
            status = decisions[(case["case"], claim)]
            x = left + column_index * cell_w
            draw.rounded_rectangle(
                (x + 8, y + 8, x + cell_w - 8, y + cell_h - 8),
                radius=12,
                fill=colors[status],
                outline=WHITE,
                width=3,
            )
            short_status = {
                "supported": "PASS",
                "blocked": "BLOCK",
                "uncertain": "UNKNOWN",
                "out_of_scope": "N/A",
                "needs_external_review": "REVIEW",
            }[status]
            draw.text(
                (x + cell_w // 2, y + cell_h // 2),
                short_status,
                font=SMALL_BOLD,
                fill=WHITE if status != "out_of_scope" else INK,
                anchor="mm",
            )
    legend = [
        ("supported", "Supported"),
        ("blocked", "Blocked"),
        ("uncertain", "Unknown evidence"),
        ("out_of_scope", "Not applicable"),
        ("needs_external_review", "External review"),
    ]
    legend_x = 380
    legend_y = 760
    for status, label in legend:
        draw.rounded_rectangle(
            (legend_x, legend_y, legend_x + 42, legend_y + 42),
            radius=7,
            fill=colors[status],
        )
        draw.text((legend_x + 55, legend_y + 21), label, font=SMALL, fill=INK, anchor="lm")
        legend_x += 340
    _wrapped(
        draw,
        (1100, 845),
        "Internal reproduction is within a resource. It is not external replication.",
        font=SMALL,
        fill=MUTED,
        width=85,
    )
    return img


def _rate_x(value: float, x0: int, x1: int, maximum: float) -> int:
    return x0 + int((x1 - x0) * value / maximum)


def build_oracle_benchmark() -> Image.Image:
    """Plot oracle error rates and paired policy correctness."""

    payload = json.loads(
        (ROOT / "results" / "oracle_sem_claim_benchmark" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    rows = {row["policy"]: row for row in payload["aggregate_by_policy"]}
    img = Image.new("RGB", (1800, 780), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((450, 55), "A. Error rates over 5,000 claim decisions", font=HEAD, fill=INK, anchor="mm")
    x0, x1 = 210, 800
    maximum = 0.30
    for tick in range(4):
        value = tick * 0.10
        x = _rate_x(value, x0, x1, maximum)
        draw.line((x, 115, x, 610), fill="#d1d5db", width=2)
        draw.text((x, 635), f"{value:.2f}", font=SMALL, fill=MUTED, anchor="ma")
    policies = [
        ("Evidence contract v3", "evidence_contract_v3", "#447ba6"),
        ("Prediction shortcut", "prediction_shortcut", "#b96a5e"),
    ]
    metrics = [
        ("False-positive rate", "false_positive_rate", "false_positive_rate_wilson_95"),
        ("False-negative rate", "false_negative_rate", "false_negative_rate_wilson_95"),
    ]
    y = 150
    for policy_label, policy_key, color in policies:
        draw.text((80, y + 50), policy_label, font=BODY, fill=INK, anchor="lm")
        for metric_label, metric_key, interval_key in metrics:
            row = rows[policy_key]
            value = row[metric_key]
            low, high = row[interval_key]
            draw.text((190, y + 102), metric_label, font=SMALL, fill=MUTED, anchor="rm")
            bar_end = _rate_x(value, x0, x1, maximum)
            draw.rounded_rectangle((x0, y + 82, bar_end, y + 118), radius=8, fill=color)
            low_x = _rate_x(low, x0, x1, maximum)
            high_x = _rate_x(high, x0, x1, maximum)
            draw.line((low_x, y + 100, high_x, y + 100), fill=INK, width=4)
            draw.line((low_x, y + 91, low_x, y + 109), fill=INK, width=3)
            draw.line((high_x, y + 91, high_x, y + 109), fill=INK, width=3)
            label_x = max(bar_end, high_x) + 22
            draw.text((label_x, y + 100), f"{value:.3f}", font=SMALL_BOLD, fill=INK, anchor="lm")
            y += 70
        y += 40

    draw.line((900, 35, 900, 720), fill="#94a3b8", width=3)
    draw.text((1350, 55), "B. Paired decision correctness", font=HEAD, fill=INK, anchor="mm")
    paired = payload["paired_decision_counts"]
    paired_rows = [
        ("Both correct", paired["both_correct"], "#6b9f7c"),
        ("Contract only correct", paired["contract_only_correct"], "#447ba6"),
        ("Shortcut only correct", paired["shortcut_only_correct"], "#d59b45"),
        ("Both wrong", paired["both_wrong"], "#b96a5e"),
    ]
    max_count = max(value for _label, value, _color in paired_rows)
    for index, (label, value, color) in enumerate(paired_rows):
        y = 145 + index * 115
        draw.text((1000, y + 25), label, font=BODY, fill=INK, anchor="lm")
        draw.rounded_rectangle((1000, y + 55, 1690, y + 95), radius=9, fill="#e5e7eb")
        end = 1000 + int(690 * value / max_count)
        draw.rounded_rectangle((1000, y + 55, end, y + 95), radius=9, fill=color)
        draw.text((end + 14, y + 75), str(value), font=SMALL_BOLD, fill=INK, anchor="lm")
    case_level = payload["case_level_policy_comparison"]
    _wrapped(
        draw,
        (1350, 650),
        "Case-level comparison: "
        f"contract better in {case_level['contract_fewer_errors']}, "
        f"shortcut better in {case_level['shortcut_fewer_errors']}, "
        f"ties in {case_level['tied_errors']}. "
        f"Exact sign-test p = {case_level['exact_two_sided_sign_test_p_value']:.2e}.",
        font=SMALL,
        fill=MUTED,
        width=70,
    )
    return img


def build_external_controls() -> Image.Image:
    """Plot SciFact decision trade-offs and Tuebingen directional uncertainty."""

    scifact = json.loads(
        (ROOT / "results" / "scifact_claim_verification" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    tuebingen = json.loads(
        (ROOT / "results" / "tuebingen_causal_direction" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    img = Image.new("RGB", (1800, 830), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((450, 55), "A. SciFact support decisions", font=HEAD, fill=INK, anchor="mm")
    policies = [
        (
            "Cited-text shortcut",
            scifact["shortcut_overclaiming_risk"],
            scifact["shortcut_conservativeness"],
        ),
        (
            "BM25 + rationale",
            scifact["retrieval_overclaiming_risk"],
            scifact["retrieval_conservativeness"],
        ),
        (
            "Train-calibrated",
            scifact["calibrated_false_positive_rate"],
            scifact["calibrated_false_negative_rate"],
        ),
    ]
    x0, x1 = 270, 810
    for tick in range(5):
        value = tick * 0.20
        x = _rate_x(value, x0, x1, 0.80)
        draw.line((x, 115, x, 635), fill="#d1d5db", width=2)
        draw.text((x, 660), f"{value:.1f}", font=SMALL, fill=MUTED, anchor="ma")
    for index, (label, fpr, fnr) in enumerate(policies):
        y = 145 + index * 165
        draw.text((80, y + 55), label, font=BODY, fill=INK, anchor="lm")
        for offset, metric, value, color in (
            (75, "FPR", fpr, "#447ba6"),
            (125, "FNR", fnr, "#b96a5e"),
        ):
            draw.text((250, y + offset), metric, font=SMALL_BOLD, fill=MUTED, anchor="rm")
            end = _rate_x(value, x0, x1, 0.80)
            draw.rounded_rectangle((x0, y + offset - 17, end, y + offset + 17), radius=7, fill=color)
            draw.text((end + 12, y + offset), f"{value:.3f}", font=SMALL_BOLD, fill=INK, anchor="lm")

    draw.line((900, 35, 900, 755), fill="#94a3b8", width=3)
    draw.text((1350, 55), "B. Tuebingen direction control", font=HEAD, fill=INK, anchor="mm")
    draw.text((1010, 150), "Unweighted accuracy", font=BODY, fill=INK)
    accuracy = tuebingen["direction_accuracy"]
    low, high = tuebingen["direction_accuracy_wilson_95"]
    tx0, tx1 = 1040, 1660
    bar_y = 230
    draw.rounded_rectangle((tx0, bar_y, tx1, bar_y + 58), radius=10, fill="#e5e7eb")
    draw.rounded_rectangle((tx0, bar_y, _rate_x(accuracy, tx0, tx1, 1.0), bar_y + 58), radius=10, fill="#447ba6")
    draw.line((_rate_x(low, tx0, tx1, 1.0), bar_y + 29, _rate_x(high, tx0, tx1, 1.0), bar_y + 29), fill=INK, width=5)
    draw.text((1350, 330), f"{accuracy:.3f}  (95% CI {low:.3f}--{high:.3f})", font=HEAD, fill=INK, anchor="mm")
    draw.text((1010, 430), "Attempted direction", font=BODY, fill=INK)
    draw.text((1650, 430), f"103 / 108 pairs ({tuebingen['direction_attempt_rate']:.1%})", font=BODY, fill=INK, anchor="ra")
    draw.text((1010, 510), "Correlation-only overclaims", font=BODY, fill=INK)
    draw.text((1650, 510), f"{tuebingen['correlation_only_direction_overclaims']} / 108 pairs", font=BODY, fill=INK, anchor="ra")
    _wrapped(
        draw,
        (1350, 630),
        "The directional baseline is not competitive. Its role is to expose why association must not authorize causal direction.",
        font=BODY,
        fill=MUTED,
        width=58,
    )
    return img


def build_dashboard() -> Image.Image:
    """Create the evidence dashboard figure from frozen manuscript results."""
    extraction = json.loads(
        (ROOT / "results" / "llm_claim_extraction_audit" / "summary.json").read_text(encoding="utf-8")
    )
    reproduction = json.loads(
        (ROOT / "results" / "claimbench_reproduction_manifest" / "summary.json").read_text(encoding="utf-8")
    )
    unified = json.loads(
        (ROOT / "results" / "claimbench_unified_report" / "summary.json").read_text(encoding="utf-8")
    )
    threat_model = json.loads(
        (ROOT / "results" / "claimbench_threat_model" / "summary.json").read_text(encoding="utf-8")
    )
    release = json.loads(
        (ROOT / "results" / "claimbench_v2_release" / "summary.json").read_text(encoding="utf-8")
    )

    img = Image.new("RGB", (1800, 835), WHITE)
    draw = ImageDraw.Draw(img)

    clean = not release["missing_artifacts"] and not release["dirty_artifacts"] and not release["failing_artifacts"]
    cards = [
        (80, 35, 460, 215, "Reproduction", f"{reproduction['passed_stages']} / {reproduction['num_stages']} stages", GREEN),
        (500, 35, 880, 215, "Unified criteria", f"{unified['passed_criteria']} / {unified['num_criteria']} passed", BLUE),
        (920, 35, 1300, 215, "Threat model", f"{threat_model['passed_threats']} / {threat_model['num_threats']} passed", AMBER),
        (1340, 35, 1720, 215, "Release state", "Clean" if clean else "Action required", GREEN if clean else RED),
    ]
    for x0, y0, x1, y1, title, body, fill in cards:
        _box(draw, (x0, y0, x1, y1), title, body, fill=fill)

    draw.text((80, 280), "Candidate statements by claim family", font=HEAD, fill=INK)
    counts = extraction["candidate_counts_by_type"]
    display = [
        ("Mechanistic", counts["mechanistic"]),
        ("Prediction", counts["prediction"]),
        ("Causal", counts["causal"]),
        ("Digital twin", counts["digital_twin"]),
        ("Structure-function", counts["structure_function"]),
        ("Reproducibility", counts["reproducibility"]),
        ("Generalization", counts["generalization"]),
        ("State of the art", counts["sota"]),
    ]
    max_count = max(value for _, value in display)
    chart_x0, chart_x1 = 390, 1660
    for index, (label, value) in enumerate(display):
        y = 335 + index * 61
        draw.text((80, y + 16), label, font=BODY, fill=INK, anchor="lm")
        draw.rounded_rectangle((chart_x0, y, chart_x1, y + 34), radius=8, fill="#e5e7eb")
        bar_end = chart_x0 + int((chart_x1 - chart_x0) * value / max_count)
        draw.rounded_rectangle((chart_x0, y, bar_end, y + 34), radius=8, fill="#4f81a8")
        draw.text((bar_end + 14, y + 17), str(value), font=SMALL_BOLD, fill=INK, anchor="lm")

    return img


def build_ablation() -> Image.Image:
    """Create the legacy conformance-ablation figure using standard FPR."""
    payload = json.loads((ROOT / "results" / "claim_adversarial_v2" / "summary.json").read_text(encoding="utf-8"))
    by_name = {row["evaluator"]: row for row in payload["aggregate_by_evaluator"]}
    evaluators = [
        ("Correlation only", "correlation_only"),
        ("Leaderboard only", "leaderboard_only"),
        ("Reliability only", "reliability_only"),
        ("Compensatory score", "compensatory_score"),
        ("Topology only", "topology_only"),
        ("No topology block", "ablated_claim_gate_no_topology"),
        ("No direction block", "ablated_claim_gate_no_directed"),
        ("Full ClaimBench gate", "claim_gate"),
    ]

    img = Image.new("RGB", (1800, 800), WHITE)
    draw = ImageDraw.Draw(img)

    chart_x0, chart_x1 = 500, 1660
    chart_y0, chart_y1 = 55, 695
    max_x = 0.55
    for tick in range(0, 12):
        value = tick * 0.05
        x = chart_x0 + int((chart_x1 - chart_x0) * value / max_x)
        draw.line((x, chart_y0, x, chart_y1), fill="#d1d5db", width=2)
        draw.text((x, chart_y1 + 24), f"{value:.2f}", font=SMALL, fill=MUTED, anchor="ma")

    for index, (label, key) in enumerate(evaluators):
        row = by_name[key]
        value = row["false_positive_rate"]
        y = chart_y0 + index * 76
        draw.text((470, y + 24), label, font=BODY, fill=INK, anchor="rm")
        bar_end = chart_x0 + int((chart_x1 - chart_x0) * value / max_x)
        color = "#b64b4b" if value >= 0.10 else "#d69b3b" if value > 0 else "#3d8b63"
        if value == 0:
            draw.ellipse((chart_x0 - 7, y + 13, chart_x0 + 7, y + 27), fill=color)
        else:
            draw.rounded_rectangle((chart_x0, y + 6, bar_end, y + 34), radius=7, fill=color)
        draw.text((max(bar_end + 14, chart_x0 + 18), y + 20), f"{value:.3f}", font=SMALL_BOLD, fill=INK, anchor="lm")

    return img


def build_sensitivity() -> Image.Image:
    """Create threshold-sensitivity and deterministic stability panels."""
    threshold = json.loads(
        (ROOT / "results" / "claim_threshold_sensitivity_v2" / "summary.json").read_text(encoding="utf-8")
    )
    uncertainty = json.loads(
        (ROOT / "results" / "uncertainty_claim_gate_v2" / "summary.json").read_text(encoding="utf-8")
    )
    img = Image.new("RGB", (1800, 700), WHITE)
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle((70, 35, 860, 665), radius=18, fill=WHITE, outline="#94a3b8", width=3)
    draw.text((465, 80), "A. Threshold grid", font=HEAD, fill=INK, anchor="mm")
    total = threshold["num_threshold_cells"]
    safe = threshold["safe_cells"]
    dangerous = threshold["dangerous_cells"]
    bar_x0, bar_x1 = 145, 785
    split = bar_x0 + int((bar_x1 - bar_x0) * safe / total)
    draw.rounded_rectangle((bar_x0, 175, bar_x1, 255), radius=14, fill="#e5e7eb")
    draw.rounded_rectangle((bar_x0, 175, split, 255), radius=14, fill="#3d8b63")
    draw.rounded_rectangle((split, 175, bar_x1, 255), radius=14, fill="#b64b4b")
    draw.text(((bar_x0 + split) // 2, 215), f"Safe\n{safe}", font=SMALL_BOLD, fill=WHITE, anchor="mm", align="center")
    draw.text(((split + bar_x1) // 2, 215), f"Dangerous\n{dangerous}", font=SMALL_BOLD, fill=WHITE, anchor="mm", align="center")
    draw.text((465, 325), f"{safe / total:.1%} safe  |  {dangerous / total:.1%} dangerous", font=HEAD, fill=INK, anchor="mm")
    _wrapped(
        draw,
        (465, 395),
        f"{total} combinations around five nominal thresholds. Dangerous means at least one unsupported claim was authorized.",
        font=BODY,
        fill=MUTED,
        width=52,
    )

    draw.rounded_rectangle((940, 35, 1730, 665), radius=18, fill=WHITE, outline="#94a3b8", width=3)
    draw.text((1335, 80), "B. Three-state gate", font=HEAD, fill=INK, anchor="mm")
    statuses = [
        ("Supported", uncertainty["status_counts"]["supported"], "#3d8b63"),
        ("Uncertain", uncertainty["status_counts"]["uncertain"], "#d69b3b"),
        ("Blocked", uncertainty["status_counts"]["blocked"], "#b64b4b"),
    ]
    max_status = max(value for _, value, _ in statuses)
    for index, (label, value, color) in enumerate(statuses):
        y = 175 + index * 105
        draw.text((1060, y + 23), label, font=BODY, fill=INK, anchor="rm")
        draw.rounded_rectangle((1100, y, 1620, y + 46), radius=9, fill="#e5e7eb")
        bar_end = 1100 + int(520 * value / max_status)
        draw.rounded_rectangle((1100, y, bar_end, y + 46), radius=9, fill=color)
        draw.text((bar_end + 16, y + 23), str(value), font=SMALL_BOLD, fill=INK, anchor="lm")
    _wrapped(
        draw,
        (1335, 520),
        f"0 unsupported claims marked supported. {uncertainty['supported_uncertain']} supportable claims become uncertain under local perturbations.",
        font=BODY,
        fill=MUTED,
        width=52,
    )
    return img


def write_all() -> None:
    """Write all versioned PNG assets to both manuscript roots."""
    figures = {
        "claimbench_workflow.png": build_workflow(),
        "claimbench_real_case_matrix.png": build_real_case_matrix(),
        "claimbench_oracle_benchmark.png": build_oracle_benchmark(),
        "claimbench_external_controls.png": build_external_controls(),
        "claimbench_ablation.png": build_ablation(),
        "claimbench_sensitivity.png": build_sensitivity(),
    }
    for directory in FIGURE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        for filename, image in figures.items():
            image.save(directory / filename, optimize=True)


if __name__ == "__main__":
    write_all()

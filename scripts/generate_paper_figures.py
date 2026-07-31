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


TITLE = _font(36, bold=True)
SUBTITLE = _font(23)
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
    img = Image.new("RGB", (1800, 820), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((900, 36), "Claim-aware decision-support workflow", font=TITLE, fill=INK, anchor="mm")
    draw.text(
        (900, 82),
        "Candidate claims are separated from claim authorization, and every decision points to an executable artifact.",
        font=SUBTITLE,
        fill=MUTED,
        anchor="mm",
    )

    boxes = [
        (55, 155, 345, 405, "Source package", "Manuscript\nfrozen results\nrelease metadata", BLUE),
        (405, 155, 695, 405, "Claim discovery", "Rules or LLMs\npropose candidates\nwithout authority", GREY),
        (755, 155, 1045, 405, "Claim contract", "Required blocks\nscope and thresholds\nadmissible wording", GREEN),
        (1105, 155, 1395, 405, "Artifact checks", "Metrics and controls\nprovenance\nrelease integrity", BLUE),
        (1455, 155, 1745, 405, "Gate decision", "Supported\nblocked\nuncertain", AMBER),
    ]
    for x0, y0, x1, y1, title, body, fill in boxes:
        _box(draw, (x0, y0, x1, y1), title, body, fill=fill)
    for start, end in [
        ((345, 280), (405, 280)),
        ((695, 280), (755, 280)),
        ((1045, 280), (1105, 280)),
        ((1395, 280), (1455, 280)),
    ]:
        _arrow(draw, start, end)

    _box(
        draw,
        (125, 510, 725, 690),
        "Bounded wording retained",
        "Every mandatory block passes for the declared claim scope.",
        fill=GREEN,
        outline="#26734d",
    )
    _box(
        draw,
        (1075, 510, 1675, 690),
        "Wording revised or withheld",
        "A failed or unstable block remains visible and cannot be compensated.",
        fill=RED,
        outline="#a33a32",
    )
    _arrow(draw, (1600, 405), (1375, 510))
    _arrow(draw, (1600, 405), (425, 510))
    draw.rounded_rectangle((125, 735, 1675, 795), radius=16, fill="#eef2ff", outline="#6366a5", width=3)
    draw.text(
        (900, 765),
        "Human authority boundary: authors and reviewers retain the final scientific judgement",
        font=HEAD,
        fill=INK,
        anchor="mm",
    )
    return img


def _table(draw: ImageDraw.ImageDraw, origin: tuple[int, int], widths: list[int],
           rows: list[list[str]], *, header_fill: str = "#e5e7eb") -> int:
    """Draw a compact table and return its height."""
    x0, y = origin
    row_h = 74
    for r, row in enumerate(rows):
        x = x0
        fill = header_fill if r == 0 else WHITE
        max_h = row_h if r == 0 else 92
        for width, cell in zip(widths, row):
            draw.rectangle((x, y, x + width, y + max_h), fill=fill, outline="#64748b", width=2)
            font = SMALL_BOLD if r == 0 else SMALL
            _wrapped(draw, (x + 16, y + 18), cell, font=font, fill=INK if r == 0 else MUTED, width=max(12, width // 16), align="left")
            x += width
        y += max_h
    return y - origin[1]


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

    img = Image.new("RGB", (1800, 1040), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((900, 42), "Frozen release evidence and candidate-claim profile", font=TITLE, fill=INK, anchor="mm")
    draw.text(
        (900, 88),
        "Engineering integrity and text-discovery outputs are reported separately.",
        font=SUBTITLE,
        fill=MUTED,
        anchor="mm",
    )

    clean = not release["missing_artifacts"] and not release["dirty_artifacts"] and not release["failing_artifacts"]
    cards = [
        (80, 145, 460, 325, "Reproduction", f"{reproduction['passed_stages']} / {reproduction['num_stages']} stages", GREEN),
        (500, 145, 880, 325, "Unified criteria", f"{unified['passed_criteria']} / {unified['num_criteria']} passed", BLUE),
        (920, 145, 1300, 325, "Threat model", f"{threat_model['passed_threats']} / {threat_model['num_threats']} passed", AMBER),
        (1340, 145, 1720, 325, "Release state", "Clean" if clean else "Action required", GREEN if clean else RED),
    ]
    for x0, y0, x1, y1, title, body, fill in cards:
        _box(draw, (x0, y0, x1, y1), title, body, fill=fill)

    draw.text((80, 390), "Candidate statements by claim family", font=HEAD, fill=INK)
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
        y = 445 + index * 61
        draw.text((80, y + 16), label, font=BODY, fill=INK, anchor="lm")
        draw.rounded_rectangle((chart_x0, y, chart_x1, y + 34), radius=8, fill="#e5e7eb")
        bar_end = chart_x0 + int((chart_x1 - chart_x0) * value / max_count)
        draw.rounded_rectangle((chart_x0, y, bar_end, y + 34), radius=8, fill="#4f81a8")
        draw.text((bar_end + 14, y + 17), str(value), font=SMALL_BOLD, fill=INK, anchor="lm")

    draw.rounded_rectangle((80, 945, 1720, 1005), radius=14, fill=GREY, outline="#94a3b8", width=2)
    draw.text(
        (900, 975),
        f"{extraction['num_candidates']} candidates from one frozen source corpus | no LLM API call | non-authoritative output",
        font=SMALL_BOLD,
        fill=MUTED,
        anchor="mm",
    )
    return img


def build_ablation() -> Image.Image:
    """Create the ablation figure with the frozen overclaiming-risk signals."""
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

    img = Image.new("RGB", (1800, 1050), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((900, 42), "Known-truth ablation: unsupported claim authorization", font=TITLE, fill=INK, anchor="mm")
    draw.text(
        (900, 88),
        "Overclaiming Risk Index (ORI). Lower values are better, but must be read with conservativeness.",
        font=SUBTITLE,
        fill=MUTED,
        anchor="mm",
    )

    chart_x0, chart_x1 = 500, 1660
    chart_y0, chart_y1 = 190, 890
    max_x = 0.55
    for tick in range(0, 12):
        value = tick * 0.05
        x = chart_x0 + int((chart_x1 - chart_x0) * value / max_x)
        draw.line((x, chart_y0, x, chart_y1), fill="#d1d5db", width=2)
        draw.text((x, chart_y1 + 24), f"{value:.2f}", font=SMALL, fill=MUTED, anchor="ma")

    for index, (label, key) in enumerate(evaluators):
        row = by_name[key]
        value = row["overclaiming_risk_index"]
        y = chart_y0 + index * 84
        draw.text((470, y + 24), label, font=BODY, fill=INK, anchor="rm")
        bar_end = chart_x0 + int((chart_x1 - chart_x0) * value / max_x)
        color = "#b64b4b" if value >= 0.10 else "#d69b3b" if value > 0 else "#3d8b63"
        if value == 0:
            draw.ellipse((chart_x0 - 7, y + 13, chart_x0 + 7, y + 27), fill=color)
        else:
            draw.rounded_rectangle((chart_x0, y + 6, bar_end, y + 34), radius=7, fill=color)
        draw.text((max(bar_end + 14, chart_x0 + 18), y + 20), f"{value:.3f}", font=SMALL_BOLD, fill=INK, anchor="lm")

    draw.text((1080, 995), "ORI = FP / (FP + TN)", font=HEAD, fill=INK, anchor="mm")
    draw.text((1080, 1028), "The no-reproducibility ablation also produced ORI = 0.000 in this suite.", font=SMALL, fill=MUTED, anchor="mm")
    return img


def build_sensitivity() -> Image.Image:
    """Create threshold-sensitivity and deterministic stability panels."""
    threshold = json.loads(
        (ROOT / "results" / "claim_threshold_sensitivity_v2" / "summary.json").read_text(encoding="utf-8")
    )
    uncertainty = json.loads(
        (ROOT / "results" / "uncertainty_claim_gate_v2" / "summary.json").read_text(encoding="utf-8")
    )
    img = Image.new("RGB", (1800, 860), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((900, 42), "Threshold sensitivity and deterministic decision stability", font=TITLE, fill=INK, anchor="mm")
    draw.text(
        (900, 88),
        "Safe-region coverage and three-state outcomes answer different robustness questions.",
        font=SUBTITLE,
        fill=MUTED,
        anchor="mm",
    )

    draw.rounded_rectangle((70, 145, 860, 775), radius=18, fill=WHITE, outline="#94a3b8", width=3)
    draw.text((465, 190), "A. Threshold grid", font=HEAD, fill=INK, anchor="mm")
    total = threshold["num_threshold_cells"]
    safe = threshold["safe_cells"]
    dangerous = threshold["dangerous_cells"]
    bar_x0, bar_x1 = 145, 785
    split = bar_x0 + int((bar_x1 - bar_x0) * safe / total)
    draw.rounded_rectangle((bar_x0, 285, bar_x1, 365), radius=14, fill="#e5e7eb")
    draw.rounded_rectangle((bar_x0, 285, split, 365), radius=14, fill="#3d8b63")
    draw.rounded_rectangle((split, 285, bar_x1, 365), radius=14, fill="#b64b4b")
    draw.text(((bar_x0 + split) // 2, 325), f"Safe\n{safe}", font=SMALL_BOLD, fill=WHITE, anchor="mm", align="center")
    draw.text(((split + bar_x1) // 2, 325), f"Dangerous\n{dangerous}", font=SMALL_BOLD, fill=WHITE, anchor="mm", align="center")
    draw.text((465, 435), f"{safe / total:.1%} safe  |  {dangerous / total:.1%} dangerous", font=HEAD, fill=INK, anchor="mm")
    _wrapped(
        draw,
        (465, 505),
        f"{total} combinations around five nominal thresholds. Dangerous means at least one unsupported claim was authorized.",
        font=BODY,
        fill=MUTED,
        width=52,
    )

    draw.rounded_rectangle((940, 145, 1730, 775), radius=18, fill=WHITE, outline="#94a3b8", width=3)
    draw.text((1335, 190), "B. Three-state gate", font=HEAD, fill=INK, anchor="mm")
    statuses = [
        ("Supported", uncertainty["status_counts"]["supported"], "#3d8b63"),
        ("Uncertain", uncertainty["status_counts"]["uncertain"], "#d69b3b"),
        ("Blocked", uncertainty["status_counts"]["blocked"], "#b64b4b"),
    ]
    max_status = max(value for _, value, _ in statuses)
    for index, (label, value, color) in enumerate(statuses):
        y = 285 + index * 105
        draw.text((1060, y + 23), label, font=BODY, fill=INK, anchor="rm")
        draw.rounded_rectangle((1100, y, 1620, y + 46), radius=9, fill="#e5e7eb")
        bar_end = 1100 + int(520 * value / max_status)
        draw.rounded_rectangle((1100, y, bar_end, y + 46), radius=9, fill=color)
        draw.text((bar_end + 16, y + 23), str(value), font=SMALL_BOLD, fill=INK, anchor="lm")
    _wrapped(
        draw,
        (1335, 630),
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
        "claimbench_evidence_dashboard.png": build_dashboard(),
        "claimbench_ablation.png": build_ablation(),
        "claimbench_sensitivity.png": build_sensitivity(),
    }
    for directory in FIGURE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        for filename, image in figures.items():
            image.save(directory / filename, optimize=True)


if __name__ == "__main__":
    write_all()

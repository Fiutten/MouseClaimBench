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
    img = Image.new("RGB", (1800, 760), WHITE)
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
        (80, 150, 430, 390, "Manuscript and artifacts", "Draft text\nfrozen results\nrelease metadata", BLUE),
        (520, 150, 870, 390, "Candidate claims", "Prediction\nmechanism\ncausality\ndigital-twin scope", GREY),
        (960, 150, 1310, 390, "Evidence contracts", "Required evidence\nblocking rules\nallowed wording", GREEN),
        (1400, 150, 1750, 390, "Decision artifact", "Supported\nblocked\nuncertain\nwith provenance", AMBER),
    ]
    for x0, y0, x1, y1, title, body, fill in boxes:
        _box(draw, (x0, y0, x1, y1), title, body, fill=fill)
    for start, end in [((430, 270), (520, 270)), ((870, 270), (960, 270)), ((1310, 270), (1400, 270))]:
        _arrow(draw, start, end)

    _box(draw, (250, 500, 780, 690), "Admitted claim", "All mandatory evidence blocks pass for the declared scope.", fill=GREEN, outline="#26734d")
    _box(draw, (1020, 500, 1550, 690), "Blocked or partial claim", "Missing evidence remains visible and cannot be compensated by another score.", fill=RED, outline="#a33a32")
    _arrow(draw, (1575, 390), (1285, 500))
    _arrow(draw, (1575, 390), (515, 500))
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
    img = Image.new("RGB", (1800, 1220), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((900, 42), "Evidence dashboard for ClaimBench v2", font=TITLE, fill=INK, anchor="mm")
    draw.text((900, 88), "The dashboard separates release integrity from candidate-claim distribution.", font=SUBTITLE, fill=MUTED, anchor="mm")

    checks = [
        ["Audit component", "Observed result", "Decision-support interpretation"],
        ["Reproduction manifest", "15 / 15 stages", "End-to-end package executable from frozen artifacts."],
        ["Unified report", "10 / 10 criteria", "Claim-governance criteria jointly satisfied."],
        ["Threat model", "7 / 7 threats", "Reviewer-facing risks mapped to explicit boundaries."],
        ["Release check", "Clean", "No required artifact is missing, dirty, or failing."],
        ["LLM extraction", "120 candidates", "Candidate extraction active but non-authoritative."],
    ]
    _table(draw, (80, 140), [430, 270, 900], checks)

    claims = [
        ["Claim type", "Count", "Audit role"],
        ["Mechanistic", "43", "Requires topology and direction before stronger wording."],
        ["Prediction", "41", "Can be supported without becoming mechanism."],
        ["Causal", "10", "Blocked unless causal evidence is available."],
        ["Digital twin", "8", "Requires strict scope control."],
        ["Other bounded claims", "18", "Structure-function, reproducibility, generalization, and SOTA wording."],
    ]
    _table(draw, (80, 640), [430, 170, 1000], claims)
    return img


def build_ablation() -> Image.Image:
    """Create the ablation figure with the frozen overclaiming-risk signals."""
    img = Image.new("RGB", (1800, 980), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((900, 42), "Component ablation and overclaiming risk", font=TITLE, fill=INK, anchor="mm")
    draw.text((900, 88), "The ablation identifies which unsupported statements become possible when safeguards are removed.", font=SUBTITLE, fill=MUTED, anchor="mm")

    rows = [
        ["Removed component", "Risk signal", "Interpretation"],
        ["Correlation-only evaluation", "ORI = 0.521", "Ordinary correlation authorizes unsupported topology, direction, structure-function, and mechanism wording."],
        ["Weighted score", "ORI = 0.135", "Compensation still permits unsupported claims when one evidence block is missing."],
        ["Topology or direction block", "ORI = 0.019", "Mechanistic wording can pass without the required structural evidence."],
        ["Lexical shortcut", "ORI = 0.199 to 0.250", "Retrieval, rationale support, and overclaiming risk must be reported separately."],
        ["Causal abstention", "79 / 108 overclaims", "Correlation-only evidence is incorrectly promoted to causal direction."],
        ["Manuscript audit", "Qualitative risk", "Unsupported wording can re-enter the article even when numerical artifacts are correct."],
    ]
    _table(draw, (80, 150), [460, 260, 880], rows)
    return img


def build_sensitivity() -> Image.Image:
    """Create a small sensitivity summary from the frozen q1_sensitivity artifact."""
    payload = json.loads((ROOT / "results" / "q1_sensitivity" / "summary.json").read_text(encoding="utf-8"))
    img = Image.new("RGB", (1800, 760), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((900, 42), "Sensitivity and external-control summary", font=TITLE, fill=INK, anchor="mm")
    draw.text((900, 88), "The result supports a methodological benchmark claim, not a causal or SOTA claim.", font=SUBTITLE, fill=MUTED, anchor="mm")

    allen = payload["allen_vbn"]
    static = payload["sensorium_static"]
    dyn = payload["dynamic_sensorium"]["cohorts"]

    _box(draw, (80, 150, 560, 610), "Allen VBN", f"Stable negative case\n{len(allen['rows'])} threshold settings\nmechanistic identifiability remains blocked", fill=RED, outline="#a33a32")
    _box(draw, (660, 150, 1140, 610), "Static Sensorium", f"Partial positive case\n{static['stable_partial_positive']} / {static['n_checks']} settings pass\nreliability and topography separated", fill=GREEN, outline="#26734d")
    dynamic_text = (
        "Predictive evidence case\n"
        f"{dyn[0]['n_mice']} current mice and {dyn[1]['n_mice']} OOD mice\n"
        "small NN gains reported without mechanistic wording"
    )
    _box(draw, (1240, 150, 1720, 610), "Dynamic Sensorium", dynamic_text, fill=BLUE, outline="#315f7d")
    _arrow(draw, (560, 380), (660, 380))
    _arrow(draw, (1140, 380), (1240, 380))
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

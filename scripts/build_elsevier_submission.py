"""Build a flat Elsevier Editorial Manager LaTeX source bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist" / "elsevier-submission"
INPUT_PATTERN = re.compile(r"\\input\{([^}]+)\}")
GRAPHICS_PATTERN = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")


def _flatten_tex(text: str) -> str:
    """Remove authoring subdirectory prefixes from LaTeX references."""

    text = re.sub(r"\\input\{(?:sections|tables|figures)/([^}]+)\}", r"\\input{\1}", text)
    text = re.sub(
        r"\\includegraphics(\[[^]]*\])?\{figures/([^}]+)\}",
        lambda match: f"\\includegraphics{match.group(1) or ''}{{{match.group(2)}}}",
        text,
    )
    return text


def _resolve_tex_dependencies(entrypoint: Path) -> tuple[list[Path], list[Path]]:
    """Return the TeX and graphics files reachable from one entrypoint."""

    pending = [entrypoint]
    tex_sources: list[Path] = []
    graphics: list[Path] = []
    seen: set[Path] = set()
    while pending:
        source = pending.pop()
        source = source if source.suffix else source.with_suffix(".tex")
        source = source.resolve()
        if source in seen:
            continue
        if not source.is_file():
            raise FileNotFoundError(f"Missing manuscript dependency: {source}")
        seen.add(source)
        tex_sources.append(source)
        text = source.read_text()
        for name in INPUT_PATTERN.findall(text):
            dependency = ROOT / name
            pending.append(
                dependency if dependency.suffix else dependency.with_suffix(".tex")
            )
        for name in GRAPHICS_PATTERN.findall(text):
            graphic = (ROOT / name).resolve()
            if not graphic.is_file():
                raise FileNotFoundError(f"Missing manuscript graphic: {graphic}")
            graphics.append(graphic)
    return sorted(tex_sources), sorted(set(graphics))


def run(output: Path = DEFAULT_OUTPUT) -> Path:
    """Create a flat source tree and a checksum manifest."""

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    tex_sources, graphics = _resolve_tex_dependencies(ROOT / "main.tex")
    for source in tex_sources:
        (output / source.name).write_text(_flatten_tex(source.read_text()))

    for support_file in (
        "highlights.txt",
        "references.bib",
        "elsarticle.cls",
        "elsarticle-num.bst",
        "elsarticle-harv.bst",
    ):
        shutil.copy2(ROOT / support_file, output / support_file)
    for source in graphics:
        shutil.copy2(source, output / source.name)

    manifest = {}
    for path in sorted(output.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.output).resolve())}))


if __name__ == "__main__":
    main()

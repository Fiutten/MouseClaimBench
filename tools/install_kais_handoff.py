#!/usr/bin/env python3
"""Install the verified KAIS package onto this work branch without touching historical sources."""
import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile

EXPECTED_TEX_SHA256 = "40148d60556f3a7bdc6d28ba9b2c3aea141bf50f637aa9091cc63cc7f8b37f53"
CANONICAL = "MouseClaimBench_KAIS_single.tex"
REQUIRED = {
    CANONICAL, "MouseClaimBench_KAIS_single.pdf", "references.bib",
    "standards_workflow.pdf", "claim_evidence_matrix_v2.pdf",
    "integrity_ablation_v2.pdf", "scalability_v2.pdf",
    "prospective_applications_v2.pdf",
}
def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def main():
    p = argparse.ArgumentParser(description="Install author-reviewed KAIS manuscript package")
    p.add_argument("submission_zip", type=Path, help="Locally downloaded verified KAIS submission ZIP")
    p.add_argument("--dest", type=Path, default=Path("paper/kais"))
    args = p.parse_args()
    with ZipFile(args.submission_zip) as z:
        names = z.namelist()
        assert len(names) == len(set(names)), "Duplicate filenames in archive"
        assert all(Path(n).name == n for n in names), "Only flat safe archive entries permitted"
        assert REQUIRED.issubset(set(names)), "Missing required submission file(s)"
        assert [n for n in names if n.endswith(".tex")] == [CANONICAL], "Not a one-TeX submission"
        data = {n: z.read(n) for n in REQUIRED}
    assert sha256(data[CANONICAL]) == EXPECTED_TEX_SHA256, "Manuscript SHA-256 does not match reviewed source"
    args.dest.mkdir(parents=True, exist_ok=True)
    for name, b in data.items():
        target = args.dest / name
        if target.exists() and target.read_bytes() != b:
            raise RuntimeError(f"Refusing to overwrite differing file: {target}")
    for name, b in sorted(data.items()):
        (args.dest / name).write_bytes(b)
        print(f"Wrote {args.dest/name} ({len(b)} bytes)")
    print("\nVerify and commit only after co-author review:")
    print(f"  git add {args.dest}")
    print('  git commit -m "paper(kais): add verified consolidated submission"')

if __name__ == "__main__":
    main()

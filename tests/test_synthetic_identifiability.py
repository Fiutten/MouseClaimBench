import json

from mousebrainbench.benchmarks.synthetic_identifiability import run


def test_synthetic_benchmark_separates_directed_and_common_drive(tmp_path) -> None:
    output = run(tmp_path / "synthetic.json")
    payload = json.loads(output.read_text())
    cases = {case["case"]: case for case in payload["cases"]}

    assert cases["directed_truth"]["mis"]["passed"]
    assert not cases["common_drive_nonidentifiable"]["mis"]["passed"]
    assert not cases["topology_without_direction"]["mis"]["passed"]
    assert not cases["direction_without_topology_specificity"]["mis"]["passed"]

    topology_blocks = {
        block["name"]: block
        for block in cases["topology_without_direction"]["mis"]["blocks"]
    }
    direction_blocks = {
        block["name"]: block
        for block in cases["direction_without_topology_specificity"]["mis"]["blocks"]
    }

    assert topology_blocks["topology_specificity"]["passed"]
    assert not topology_blocks["directed_identifiability"]["passed"]
    assert direction_blocks["directed_identifiability"]["passed"]
    assert not direction_blocks["topology_specificity"]["passed"]

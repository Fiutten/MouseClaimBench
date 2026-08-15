"""Benchmark profile-v2 interoperability against PROV-O and pySHACL."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.profile_v2_contract_mutation import generate_cases
from mousebrainbench.knowledge import ClaimAuthorizationSystem, load_authorization_profile_v2
from mousebrainbench.knowledge.standards import (
    evidence_package_to_rdf,
    profile_to_rdf,
    shacl_shapes_for_claim,
    validate_structure_with_shacl_v2,
)

DEFAULT_PROTOCOL = Path("configs/benchmarks/profile_v2_standards.yaml")
DEFAULT_OUTPUT = Path("results/profile_v2_standards/summary.json")
DEFAULT_MARKDOWN = Path("results/profile_v2_standards/summary.md")
DEFAULT_PROFILE_RDF = Path("results/profile_v2_standards/profile.ttl")
DEFAULT_EXAMPLE_JSONLD = Path("results/profile_v2_standards/example_package.jsonld")
DEFAULT_EXAMPLE_SHAPES = Path("results/profile_v2_standards/example_shapes.ttl")


def _evaluate_case(case) -> tuple[bool, bool, bool, bool]:
    """Evaluate one picklable mutation case in an isolated SHACL worker."""

    profile = load_authorization_profile_v2()
    decision = validate_structure_with_shacl_v2(
        profile,
        case.claim,
        case.blocks,
        package_id=case.case_id,
    )
    domain = ClaimAuthorizationSystem(profile, case.blocks).infer(case.claim)
    observed_deficits = tuple(
        (row.code.value, row.witness) for row in decision.deficits
    )
    return (
        decision.conforms is case.expected_structural_conforms,
        observed_deficits == case.expected_structural_deficits,
        decision.conforms and not case.expected_structural_conforms,
        decision.conforms and not domain.authorized,
    )


def evaluate(protocol: dict[str, Any]) -> tuple[dict[str, Any], Any, Any, Any]:
    """Evaluate every mutation package with an external SHACL processor."""

    from rdflib import Graph, Namespace
    from rdflib.compare import isomorphic
    from rdflib.namespace import RDF

    profile = load_authorization_profile_v2()
    cases = generate_cases()
    expected_cases = int(protocol["case_selection"]["expected_cases"])
    if len(cases) != expected_cases:
        raise RuntimeError("standards protocol case count does not match mutation generator")
    started = time.perf_counter()
    exact = 0
    conformance_matches = 0
    false_conformances = 0
    false_nonconformances = 0
    structurally_valid_domain_refusals = 0
    workers = max(1, int(protocol.get("execution", {}).get("workers", 1)))
    if workers == 1:
        evaluated = map(_evaluate_case, cases)
    else:
        context_name = (
            "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
        )
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context(context_name),
        )
        evaluated = executor.map(_evaluate_case, cases, chunksize=8)
    try:
        for conformance_match, exact_deficits, false_conformance, domain_refusal in evaluated:
            conformance_matches += int(conformance_match)
            exact += int(exact_deficits)
            false_conformances += int(false_conformance)
            false_nonconformances += int(not conformance_match and not false_conformance)
            structurally_valid_domain_refusals += int(domain_refusal)
    finally:
        if workers > 1:
            executor.shutdown()
    elapsed = time.perf_counter() - started

    profile_graph = profile_to_rdf(profile)
    mcb = Namespace("https://w3id.org/mouseclaimbench/")
    exported_claims = len(set(profile_graph.subjects(RDF.type, mcb.ClaimType)))
    exported_blocks = len(
        set(profile_graph.subjects(RDF.type, mcb.EvidenceBlockSpecification))
    )
    example = next(case for case in cases if case.family == "pristine_complete")
    example_graph, package = evidence_package_to_rdf(
        profile,
        example.claim,
        example.blocks,
        package_id=example.case_id,
    )
    shapes = shacl_shapes_for_claim(profile, example.claim, package_node=package)
    json_ld = example_graph.serialize(format="json-ld", indent=2)
    round_trip = Graph().parse(data=json_ld, format="json-ld")
    endpoints = {
        "shacl_false_conformances_equal_0": false_conformances == 0,
        "shacl_false_nonconformances_equal_0": false_nonconformances == 0,
        "shacl_exact_structural_deficit_rate_equal_1": exact == len(cases),
        "structurally_valid_domain_refusals_present": (
            structurally_valid_domain_refusals > 0
        ),
        "prov_profile_contains_all_claims_and_blocks": (
            exported_claims == len(profile.requirements)
            and exported_blocks == len(profile.evidence_blocks)
        ),
        "json_ld_round_trip_preserves_graph_isomorphism": isomorphic(
            example_graph, round_trip
        ),
    }
    summary = {
        "profile_id": profile.profile_id,
        "profile_version": profile.version,
        "profile_hash": profile.source_hash,
        "cases": len(cases),
        "shacl": {
            "backend": f"pyshacl-{importlib.metadata.version('pyshacl')}",
            "rdflib": importlib.metadata.version("rdflib"),
            "false_conformances": false_conformances,
            "false_nonconformances": false_nonconformances,
            "conformance_matches": conformance_matches,
            "conformance_match_rate": conformance_matches / len(cases),
            "exact_structural_deficit_sets": exact,
            "exact_structural_deficit_rate": exact / len(cases),
            "structurally_valid_domain_refusals": structurally_valid_domain_refusals,
            "workers": workers,
            "elapsed_seconds": elapsed,
            "cases_per_second": len(cases) / elapsed,
        },
        "rdf": {
            "profile_triples": len(profile_graph),
            "exported_claims": exported_claims,
            "exported_evidence_blocks": exported_blocks,
            "example_package_triples": len(example_graph),
            "example_shape_triples": len(shapes),
            "json_ld_round_trip_isomorphic": isomorphic(example_graph, round_trip),
        },
        "endpoints": endpoints,
        "all_endpoints_passed": all(endpoints.values()),
        "interpretation": protocol["interpretation"],
    }
    return summary, profile_graph, example_graph, shapes


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    shacl = payload["shacl"]
    lines = [
        "# Profile v2 standards conformance",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Cases: `{payload['cases']}`",
        f"- SHACL false conformances: `{shacl['false_conformances']}`",
        f"- SHACL false non-conformances: `{shacl['false_nonconformances']}`",
        f"- Exact structural-deficit rate: `{shacl['exact_structural_deficit_rate']:.4f}`",
        f"- Structurally valid domain refusals: `{shacl['structurally_valid_domain_refusals']}`",
        f"- Throughput: `{shacl['cases_per_second']:.2f}` cases/s",
        "",
        payload["interpretation"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    profile_rdf: Path = DEFAULT_PROFILE_RDF,
    example_jsonld: Path = DEFAULT_EXAMPLE_JSONLD,
    example_shapes: Path = DEFAULT_EXAMPLE_SHAPES,
) -> Path:
    protocol = yaml.safe_load(protocol_path.read_text())
    summary, profile_graph, example_graph, shapes = evaluate(protocol)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "profile_v2_prov_o_shacl_conformance",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        **summary,
        "decision": (
            "standards_conformance_confirmed"
            if summary["all_endpoints_passed"]
            else "standards_conformance_failed"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(payload, markdown)
    profile_rdf.write_text(profile_graph.serialize(format="turtle"))
    example_jsonld.write_text(example_graph.serialize(format="json-ld", indent=2))
    example_shapes.write_text(shapes.serialize(format="turtle"))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(run(protocol_path=args.protocol, output=args.output, markdown=args.markdown).resolve())


if __name__ == "__main__":
    main()

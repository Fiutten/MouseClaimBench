#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="${ENV_PATH:-.venv-risk-v3}"
MODE="${1:-verify}"
OUT="outputs/semantic-risk-v3-reproduction"
PY="${ENV_PATH}/bin/python"

mkdir -p "${OUT}"
"${PY}" scripts/validate_semantic_risk_v3_environment.py
"${PY}" -m compileall -q mousebrainbench scripts
"${PY}" -m pytest -q

if [[ "${MODE}" == "verify" ]]; then
  "${PY}" -m mousebrainbench.benchmarks.semantic_risk_v3_release \
    --output "${OUT}/release.json" \
    --markdown "${OUT}/release.md"
  exit 0
fi

if [[ "${MODE}" != "core" && "${MODE}" != "external" ]]; then
  echo "Usage: $0 [verify|core|external]" >&2
  exit 2
fi

"${PY}" -m mousebrainbench.benchmarks.semantic_equivalence_audit \
  --output "${OUT}/semantic-equivalence.json" \
  --markdown "${OUT}/semantic-equivalence.md"
"${PY}" -m mousebrainbench.benchmarks.semantic_risk_confirmation \
  --output "${OUT}/synthetic-summary.json" \
  --markdown "${OUT}/synthetic-summary.md" \
  --cases "${OUT}/synthetic-cases.npz"

if [[ "${MODE}" == "external" ]]; then
  "${PY}" scripts/validate_semantic_risk_v3_environment.py --require-external-data
  "${PY}" -m mousebrainbench.benchmarks.causal_chambers_transport \
    --output "${OUT}/causal-chambers.json" \
    --markdown "${OUT}/causal-chambers.md"
  "${PY}" -m mousebrainbench.benchmarks.causalbench_transport \
    --output "${OUT}/causalbench.json" \
    --markdown "${OUT}/causalbench.md"
  "${PY}" -m mousebrainbench.benchmarks.ibl_mouse_transport \
    --output "${OUT}/ibl.json" \
    --markdown "${OUT}/ibl.md"
fi

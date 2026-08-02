#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="${1:-.venv-risk-v3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m venv "${ENV_PATH}"
"${ENV_PATH}/bin/python" -m pip install --upgrade pip
"${ENV_PATH}/bin/python" -m pip install -r requirements-semantic-risk-v3-lock.txt
"${ENV_PATH}/bin/python" -m pip install --no-deps -e .
"${ENV_PATH}/bin/python" scripts/validate_semantic_risk_v3_environment.py

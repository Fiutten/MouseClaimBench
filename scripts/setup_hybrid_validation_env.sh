#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="${1:-.venv-hybrid}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m venv "${ENV_PATH}"
"${ENV_PATH}/bin/python" -m pip install --upgrade pip
"${ENV_PATH}/bin/python" -m pip install -r requirements-lock.txt
"${ENV_PATH}/bin/python" -m pip install -r requirements-hybrid-lock.txt
"${ENV_PATH}/bin/python" -m pip install -e .
"${ENV_PATH}/bin/python" scripts/validate_hybrid_environment.py


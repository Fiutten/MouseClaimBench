#!/usr/bin/env bash
set -euo pipefail

revision="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked working tree must be clean before v4 reproduction." >&2
  exit 1
fi
export MOUSEBRAINBENCH_GIT_REVISION="$revision"

.venv-risk-v3/bin/python -m pytest -q
.venv-risk-v3/bin/python -m mousebrainbench.benchmarks.semantic_risk_v4_power
.venv-risk-v3/bin/python -m mousebrainbench.benchmarks.knowledge_profile_validity_v4
.venv-risk-v3/bin/python -m mousebrainbench.benchmarks.causal_chambers_v4_confirmation
.venv-risk-v3/bin/python -m mousebrainbench.benchmarks.causal_chambers_v4_1_final
.venv-risk-v3/bin/python -m mousebrainbench.benchmarks.direction_router_v4_confirmation --workers 1
.venv-risk-v3/bin/python -m mousebrainbench.benchmarks.direction_router_v4_2_confirmation --workers 1
.venv-risk-v3/bin/python -m mousebrainbench.benchmarks.semantic_risk_v4_release


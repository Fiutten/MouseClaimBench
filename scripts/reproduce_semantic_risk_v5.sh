#!/usr/bin/env bash
set -euo pipefail

env_path="${ENV_PATH:-.venv-risk-v3}"
revision="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked working tree must be clean before v5 reproduction." >&2
  exit 1
fi
export MOUSEBRAINBENCH_GIT_REVISION="$revision"

"$env_path/bin/python" scripts/fetch_timegraph_v5.py
"$env_path/bin/python" scripts/fetch_causalrivers_v5.py
"$env_path/bin/python" -m pytest -q
"$env_path/bin/python" -m mousebrainbench.benchmarks.timegraph_v5_confirmation
"$env_path/bin/python" -m mousebrainbench.benchmarks.timegraph_v5_1_topology_confirmation
"$env_path/bin/python" -m mousebrainbench.benchmarks.causalrivers_v5_transport
"$env_path/bin/python" -m mousebrainbench.benchmarks.semantic_risk_shift_sweep_v5_2
HOME="$PWD/data/external/ibl/home" \
  "$env_path/bin/python" scripts/fetch_ibl_behavior_v5.py
"$env_path/bin/python" -m mousebrainbench.benchmarks.ibl_behavior_v5_confirmation
"$env_path/bin/python" -m mousebrainbench.benchmarks.semantic_risk_orthogonal_shifts_v5_4
"$env_path/bin/python" -m mousebrainbench.benchmarks.knowledge_profile_external_validation_v5
"$env_path/bin/python" -m mousebrainbench.benchmarks.semantic_risk_v5_release

#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

env_path="${ENV_PATH:-.venv-risk-v3}"
python="$env_path/bin/python"
if [[ ! -x "$python" ]]; then
  echo "Missing Python environment: $python" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Profile v2 reproduction requires a clean working tree." >&2
  exit 2
fi

export MOUSEBRAINBENCH_GIT_REVISION="$(git rev-parse HEAD)"

"$python" -m mousebrainbench.benchmarks.profile_v2_resolution_audit
"$python" -m mousebrainbench.benchmarks.profile_v2_contract_mutation
"$python" -m mousebrainbench.benchmarks.profile_v2_artifact_application
"$python" -m mousebrainbench.benchmarks.profile_v2_release

"$python" -m compileall -q mousebrainbench scripts tests
"$python" -m pytest -q \
  tests/test_knowledge_authorization_v2.py \
  tests/test_profile_v2_resolution_audit.py \
  tests/test_profile_v2_contract_mutation.py \
  tests/test_profile_v2_artifact_application.py \
  tests/test_profile_v2_release.py

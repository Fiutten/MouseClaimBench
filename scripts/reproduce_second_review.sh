#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

mode="${1:-verify}"
env_path="${ENV_PATH:-.venv-risk-v3}"
python="$env_path/bin/python"
if [[ ! -x "$python" ]]; then
  echo "Missing Python environment: $python" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Second-review reproduction requires a clean working tree." >&2
  exit 2
fi

export MOUSEBRAINBENCH_GIT_REVISION="$(git rev-parse HEAD)"

if [[ "$mode" == "rebuild" ]]; then
  "$python" -m mousebrainbench.benchmarks.profile_v2_contract_mutation
  "$python" -m mousebrainbench.benchmarks.dandi_threshold_sensitivity
  "$python" -m mousebrainbench.benchmarks.standards_prospective_release
  "$python" -m mousebrainbench.benchmarks.profile_v2_major_revision_release
  "$python" -m mousebrainbench.benchmarks.profile_v2_second_review_release
elif [[ "$mode" != "verify" ]]; then
  echo "Usage: $0 [verify|rebuild]" >&2
  exit 2
fi

"$python" -m compileall -q mousebrainbench scripts tests
"$python" -m pytest -q \
  tests/test_profile_v2_contract_mutation.py \
  tests/test_dandi_threshold_sensitivity.py \
  tests/test_profile_v2_second_review_release.py \
  tests/test_manuscript_quality.py

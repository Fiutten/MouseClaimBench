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
  echo "Consistency-release reproduction requires a clean working tree." >&2
  exit 2
fi

export MOUSEBRAINBENCH_GIT_REVISION="$(git rev-parse HEAD)"

if [[ "$mode" == "rebuild" ]]; then
  "$python" -m mousebrainbench.benchmarks.profile_v2_contract_mutation
  "$python" -m mousebrainbench.benchmarks.profile_v2_standards
  "$python" -m mousebrainbench.benchmarks.profile_v2_provenance_attacks
  "$python" -m mousebrainbench.benchmarks.profile_v2_compositional_integrity_stress
  "$python" -m mousebrainbench.benchmarks.profile_v2_integrity_regression
  "$python" -m mousebrainbench.benchmarks.profile_v2_final_gate
  "$python" -m mousebrainbench.benchmarks.profile_v2_scalability_ablation
  "$python" -m mousebrainbench.benchmarks.profile_v2_consistency_release
elif [[ "$mode" != "verify" ]]; then
  echo "Usage: $0 [verify|rebuild]" >&2
  exit 2
fi

"$python" -m compileall -q mousebrainbench scripts tests
"$python" -m pytest -q \
  tests/test_profile_v2_contract_mutation.py \
  tests/test_profile_v2_standards.py \
  tests/test_profile_v2_provenance_attacks.py \
  tests/test_profile_v2_compositional_integrity_stress.py \
  tests/test_profile_v2_integrity_regression.py \
  tests/test_final_authorization.py \
  tests/test_profile_v2_consistency_release.py \
  tests/test_manuscript_quality.py

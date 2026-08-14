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
  echo "Major-revision reproduction requires a clean working tree." >&2
  exit 2
fi

export MOUSEBRAINBENCH_GIT_REVISION="$(git rev-parse HEAD)"

if [[ "$mode" == "rebuild" ]]; then
  "$python" -m mousebrainbench.benchmarks.profile_v2_traceability
  "$python" -m mousebrainbench.benchmarks.profile_v2_structural_sensitivity
  "$python" -m mousebrainbench.benchmarks.profile_v2_explanation_fidelity
  "$python" -m mousebrainbench.benchmarks.profile_v2_compositional_integrity_stress
  "$python" -m mousebrainbench.benchmarks.profile_v2_major_revision_release
elif [[ "$mode" != "verify" ]]; then
  echo "Usage: $0 [verify|rebuild]" >&2
  exit 2
fi

"$python" -m compileall -q mousebrainbench scripts tests
"$python" -m pytest -q \
  tests/test_profile_v2_traceability.py \
  tests/test_profile_v2_structural_sensitivity.py \
  tests/test_profile_v2_explanation_fidelity.py \
  tests/test_profile_v2_compositional_integrity_stress.py \
  tests/test_profile_v2_major_revision_release.py

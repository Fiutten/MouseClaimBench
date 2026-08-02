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
  echo "Standards/prospective reproduction requires a clean working tree." >&2
  exit 2
fi

export MOUSEBRAINBENCH_GIT_REVISION="$(git rev-parse HEAD)"

if [[ "$mode" == "rebuild" ]]; then
  "$python" -m mousebrainbench.benchmarks.profile_v2_contract_mutation
  "$python" -m mousebrainbench.benchmarks.profile_v2_standards
  "$python" -m mousebrainbench.benchmarks.profile_v2_formal_properties
  "$python" -m mousebrainbench.benchmarks.profile_v2_provenance_attacks
  "$python" -m mousebrainbench.benchmarks.profile_v2_artifact_application
  "$python" -m mousebrainbench.benchmarks.dandi_profile_v2_1
  "$python" -m mousebrainbench.benchmarks.profile_v2_scalability_ablation
  "$python" -m mousebrainbench.benchmarks.standards_prospective_release
elif [[ "$mode" != "verify" ]]; then
  echo "Usage: $0 [verify|rebuild]" >&2
  exit 2
fi

"$python" -m compileall -q mousebrainbench scripts tests
"$python" -m pytest -q \
  tests/test_knowledge_authorization_v2.py \
  tests/test_profile_v2_contract_mutation.py \
  tests/test_profile_v2_standards.py \
  tests/test_profile_v2_formal_properties.py \
  tests/test_profile_v2_provenance_attacks.py \
  tests/test_profile_v2_scalability_ablation.py \
  tests/test_profile_v2_artifact_application.py \
  tests/test_dandi_profile_v2_1.py \
  tests/test_standards_prospective_release.py

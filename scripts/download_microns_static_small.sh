#!/usr/bin/env bash
set -euo pipefail

# Download only small MICrONS static files needed for the bounded pilot gate.
# This intentionally excludes the large synapse graph; real structural edges
# should be queried through CAVE with scripts/query_microns_cave_pilot_edges.py.

ROOT="${1:-data/microns/static_small}"
mkdir -p "${ROOT}"

curl -L -C - \
  -o "${ROOT}/func_unit_em_match_release.csv" \
  "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_coregistration/func_unit_em_match_release.csv"

curl -L -C - \
  -o "${ROOT}/proofreading_status_public_release.csv" \
  "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/proofreading_status/proofreading_status_public_release.csv"

curl -L -C - \
  -o "${ROOT}/digital_twin_README.md" \
  "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/digital_twin_properties/v2/README.md"

curl -L -C - \
  -o "${ROOT}/dt_anatomy_units.csv" \
  "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/digital_twin_properties/v2/anatomy/units.csv"

curl -L -C - \
  -o "${ROOT}/dt_performance_units.csv" \
  "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/digital_twin_properties/v2/performance/units.csv"

curl -L -C - \
  -o "${ROOT}/dt_ori_dir_tuning_units.csv" \
  "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/functional_data/digital_twin_properties/v2/ori_dir_tuning/units.csv"

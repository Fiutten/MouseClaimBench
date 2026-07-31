#!/usr/bin/env bash
set -euo pipefail

# Rebuild the isolated official Sensorium 2023 smoke-test environment.
#
# This script intentionally installs the package from the official pyproject
# first, then overlays the exact neuralpredictors source commit referenced by
# the official requirements file. The full requirements.txt currently pins
# pandas==2.0.0, which is not reproducible on Python 3.12/macOS arm64 in this
# project environment.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p external .cache/matplotlib .cache/xdg

curl -L -o /tmp/sensorium_2023_main.zip \
  https://github.com/ecker-lab/sensorium_2023/archive/refs/heads/main.zip
rm -rf external/sensorium_2023 external/sensorium_2023-main
unzip -q /tmp/sensorium_2023_main.zip -d external
mv external/sensorium_2023-main external/sensorium_2023

curl -L -o /tmp/neuralpredictors_43fa.zip \
  https://github.com/sinzlab/neuralpredictors/archive/43faededa2d2e76bb904f38a49b9d8b81d287a0a.zip
rm -rf external/neuralpredictors_43fa external/neuralpredictors-43faededa2d2e76bb904f38a49b9d8b81d287a0a
unzip -q /tmp/neuralpredictors_43fa.zip -d external
mv external/neuralpredictors-43faededa2d2e76bb904f38a49b9d8b81d287a0a external/neuralpredictors_43fa

.venv/bin/python -m venv --system-site-packages .venv-sensorium-official
.venv-sensorium-official/bin/pip install --upgrade pip setuptools wheel
.venv-sensorium-official/bin/pip install -e external/sensorium_2023
.venv-sensorium-official/bin/pip install \
  "neuralpredictors @ git+https://github.com/sinzlab/neuralpredictors@43faededa2d2e76bb904f38a49b9d8b81d287a0a"
.venv-sensorium-official/bin/pip install torchvision datajoint==0.14.1 GitPython==3.1.31

MPLBACKEND=Agg \
MPLCONFIGDIR=.cache/matplotlib \
XDG_CACHE_HOME=.cache/xdg \
PYTHONPATH=external/neuralpredictors_43fa:external/sensorium_2023 \
  .venv-sensorium-official/bin/python scripts/smoke_sensorium_official_model.py

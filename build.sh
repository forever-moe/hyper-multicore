#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
# Explicit native build and wheel packaging; never turn component failure into success.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    echo 'Usage: bash build.sh --soc-list ascend910b[,ascend910_93] --jobs <positive integer> [--clean]'
    exit 0
fi
cd "${PROJECT_ROOT}"
[[ "$(uname -s)" == Linux ]] || { echo 'ERROR: native builds require Linux/CANN/PTA.' >&2; exit 3; }
PYTHON_BIN=$(command -v python)
NATIVE_ROOT="${PROJECT_ROOT}/build/native"
mkdir -p "${NATIVE_ROOT}/logs"
bash hyper_multicore/build.sh --stage-only "$@" 2>&1 | tee "${NATIVE_ROOT}/logs/multicore.log"
export HYPER_MULTICORE_NATIVE_OUTPUT_ROOT="${NATIVE_ROOT}/components/multicore/hyper_multicore"
"${PYTHON_BIN}" hyper_multicore/_build/artifacts.py --validate --output "${HYPER_MULTICORE_NATIVE_OUTPUT_ROOT}"
PAYLOAD_ROOT="${NATIVE_ROOT}/payload/hyper_multicore"
rm -rf "${PAYLOAD_ROOT}"
mkdir -p "${PAYLOAD_ROOT}"
cp -a "${HYPER_MULTICORE_NATIVE_OUTPUT_ROOT}/." "${PAYLOAD_ROOT}/"
export HYPER_MULTICORE_NATIVE_OUTPUT_ROOT="${PAYLOAD_ROOT}"
WHEEL_OUTPUT=$(mktemp -d "${NATIVE_ROOT}/wheel-output.XXXXXX")
"${PYTHON_BIN}" setup.py -q bdist_wheel --dist-dir "${WHEEL_OUTPUT}"
mapfile -t WHEELS < <(find "${WHEEL_OUTPUT}" -maxdepth 1 -type f -name 'hyper_multicore-*.whl')
[[ ${#WHEELS[@]} -eq 1 ]] || { echo 'ERROR: expected exactly one native wheel.' >&2; exit 1; }
mkdir -p dist
DESTINATION="${PROJECT_ROOT}/dist/$(basename "${WHEELS[0]}")"
if [[ -f "${DESTINATION}" ]] && ! cmp -s "${WHEELS[0]}" "${DESTINATION}"; then
    echo "ERROR: refusing to overwrite a different artifact: ${DESTINATION}" >&2
    exit 1
fi
cp -a "${WHEELS[0]}" "${DESTINATION}"
echo "Wheel: ${DESTINATION}"

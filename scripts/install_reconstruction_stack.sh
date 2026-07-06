#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PIP_PROGRESS_BAR="${PIP_PROGRESS_BAR:-off}"

log() {
  printf '\n[%(%H:%M:%S)T] %s\n' -1 "$*"
}

run() {
  log "$*"
  "$@"
}

trap 'log "FAILED at line $LINENO: $BASH_COMMAND"' ERR

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/externals"

log "Repository root: $ROOT"
run python -m pip install -r "$ROOT/requirements.txt"

INSTALL_TORCH="${INSTALL_TORCH:-auto}"
if [ "$INSTALL_TORCH" = "auto" ]; then
  if python -c "import torch" >/dev/null 2>&1; then
    log "Using existing PyTorch installation."
  else
    run python -m pip install -r "$ROOT/requirements-torch-cu118.txt"
  fi
elif [ "$INSTALL_TORCH" != "0" ]; then
  run python -m pip install -r "$ROOT/requirements-torch-cu118.txt"
else
  log "Skipping PyTorch installation because INSTALL_TORCH=0."
fi

if command -v gcc >/dev/null 2>&1; then
  export CC="${CC:-$(command -v gcc)}"
fi
if command -v g++ >/dev/null 2>&1; then
  export CXX="${CXX:-$(command -v g++)}"
fi
log "Using CC=${CC:-unset} CXX=${CXX:-unset}"

run python -m pip install ninja jaxtyping rich
GSPLAT_WHEEL_INDEX="$(
  python - <<'PY'
import platform
import re
import sys

import torch

torch_match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
cuda = (torch.version.cuda or "").replace(".", "")

if (
    sys.platform.startswith("linux")
    and platform.machine() == "x86_64"
    and torch_match
    and cuda
):
    torch_tag = "".join(torch_match.groups())
    print(f"https://docs.gsplat.studio/whl/pt{torch_tag}cu{cuda}")
PY
)"
if [ -n "$GSPLAT_WHEEL_INDEX" ]; then
  log "Trying precompiled gsplat wheel from $GSPLAT_WHEEL_INDEX"
  if python -m pip install -r "$ROOT/requirements-gsplat.txt" --index-url "$GSPLAT_WHEEL_INDEX"; then
    log "Installed gsplat from a precompiled wheel."
  else
    log "No compatible precompiled gsplat wheel installed; falling back to PyPI."
    run python -m pip install -r "$ROOT/requirements-gsplat.txt"
  fi
else
  log "No published precompiled gsplat wheel matches this Python/PyTorch/CUDA runtime; installing from PyPI."
  run python -m pip install -r "$ROOT/requirements-gsplat.txt"
fi

if [ ! -d "$ROOT/externals/Depth-Anything-3/.git" ]; then
  run git clone --depth 1 --recursive --shallow-submodules https://github.com/ByteDance-Seed/Depth-Anything-3.git "$ROOT/externals/Depth-Anything-3"
else
  log "Using existing Depth-Anything-3 checkout."
fi
run python -m pip install -e "$ROOT/externals/Depth-Anything-3"

if [ ! -d "$ROOT/externals/FoundationStereo/.git" ]; then
  run git clone --depth 1 https://github.com/NVlabs/FoundationStereo.git "$ROOT/externals/FoundationStereo"
else
  log "Using existing FoundationStereo checkout."
fi

log "Install script completed."

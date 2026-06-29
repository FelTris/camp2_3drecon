#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/externals"

python -m pip install -r "$ROOT/requirements.txt"

INSTALL_TORCH="${INSTALL_TORCH:-auto}"
if [ "$INSTALL_TORCH" = "auto" ]; then
  if python -c "import torch" >/dev/null 2>&1; then
    echo "Using existing PyTorch installation."
  else
    python -m pip install -r "$ROOT/requirements-torch-cu118.txt"
  fi
elif [ "$INSTALL_TORCH" != "0" ]; then
  python -m pip install -r "$ROOT/requirements-torch-cu118.txt"
fi

if command -v gcc >/dev/null 2>&1; then
  export CC="${CC:-$(command -v gcc)}"
fi
if command -v g++ >/dev/null 2>&1; then
  export CXX="${CXX:-$(command -v g++)}"
fi
python -m pip install --no-build-isolation -r "$ROOT/requirements-gsplat.txt"

if [ ! -d "$ROOT/externals/Depth-Anything-3/.git" ]; then
  git clone --depth 1 --recursive --shallow-submodules https://github.com/ByteDance-Seed/Depth-Anything-3.git "$ROOT/externals/Depth-Anything-3"
fi
python -m pip install -e "$ROOT/externals/Depth-Anything-3"

if [ ! -d "$ROOT/externals/FoundationStereo/.git" ]; then
  git clone --depth 1 https://github.com/NVlabs/FoundationStereo.git "$ROOT/externals/FoundationStereo"
fi

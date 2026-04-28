#!/usr/bin/env bash
set -euo pipefail

# GPU check + best-effort fix for Ubuntu/Debian + NVIDIA.
#
# What it targets:
# - NVIDIA driver presence (nvidia-smi)
# - Python deps for PaddleOCR GPU (paddlepaddle-gpu + paddleocr)
# - Quick runtime probe via scripts/gpu_probe_paddle.py
#
# Safety:
# - By default, it ONLY diagnoses.
# - Pass --fix to perform apt/pip installs (requires sudo).
#
# Usage:
#   ./scripts/gpu_check_and_fix_ubuntu.sh
#   ./scripts/gpu_check_and_fix_ubuntu.sh --fix
#   PYTHON=python3.11 ./scripts/gpu_check_and_fix_ubuntu.sh --fix

FIX=0
PYTHON="${PYTHON:-python3}"

for arg in "${@:-}"; do
  case "$arg" in
    --fix) FIX=1 ;;
    --python=*) PYTHON="${arg#--python=}" ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 64
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say() { printf "\n==> %s\n" "$*"; }
warn() { printf "\nWARNING: %s\n" "$*" >&2; }
die() { printf "\nERROR: %s\n" "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

require_sudo() {
  if [[ "$FIX" -ne 1 ]]; then
    return 0
  fi
  if ! have sudo; then
    die "sudo not found; run as root or install sudo."
  fi
  sudo -n true >/dev/null 2>&1 || {
    warn "sudo requires a password / TTY. Re-run interactively so sudo can prompt."
    die "Cannot continue in --fix mode without sudo."
  }
}

say "Environment"
echo "Repo: $repo_root"
echo "Python: $PYTHON"
echo "Fix mode: $FIX"

say "Step 1: Check NVIDIA driver (nvidia-smi)"
if have nvidia-smi; then
  nvidia-smi || warn "nvidia-smi exists but failed. Driver may be broken."
else
  warn "nvidia-smi not found -> NVIDIA driver not installed (or not in PATH)."
  if [[ "$FIX" -eq 1 ]]; then
    require_sudo
    say "Attempting to install NVIDIA driver (ubuntu-drivers autoinstall)"
    if have ubuntu-drivers; then
      sudo ubuntu-drivers autoinstall || warn "ubuntu-drivers autoinstall failed."
    else
      say "Installing ubuntu-drivers-common"
      sudo apt-get update
      sudo apt-get install -y ubuntu-drivers-common
      sudo ubuntu-drivers autoinstall || warn "ubuntu-drivers autoinstall failed."
    fi
    warn "Driver installation may require a reboot. If nvidia-smi still missing, reboot and re-run."
  fi
fi

say "Step 2: Ensure basic OS packages"
if [[ "$FIX" -eq 1 ]]; then
  require_sudo
  sudo apt-get update
  sudo apt-get install -y build-essential python3-venv python3-pip || true
fi

say "Step 3: Ensure pip + wheel"
"$PYTHON" -m pip --version >/dev/null 2>&1 || warn "pip not available for $PYTHON"
if [[ "$FIX" -eq 1 ]]; then
  "$PYTHON" -m pip install -U pip setuptools wheel
fi

say "Step 4: Install PaddleOCR GPU Python deps (best-effort)"
if [[ "$FIX" -eq 1 ]]; then
  # NOTE: Paddle GPU wheels depend on CUDA version; user may need to pick the right package index.
  # We try a generic install first; if it fails, we print guidance.
  set +e
  "$PYTHON" -m pip install -U paddleocr
  pip_rc=$?
  if [[ "$pip_rc" -ne 0 ]]; then
    warn "Failed to install paddleocr. Check Python version compatibility and network access."
  fi
  "$PYTHON" -m pip install -U paddlepaddle-gpu
  gpu_rc=$?
  set -e
  if [[ "$gpu_rc" -ne 0 ]]; then
    warn "Failed to install paddlepaddle-gpu automatically."
    warn "You likely need the correct wheel for your CUDA version. See Paddle installation guide for 'paddlepaddle-gpu'."
  fi
else
  echo "(diagnose-only) Skipping pip installs. Re-run with --fix to install paddleocr + paddlepaddle-gpu."
fi

say "Step 5: Probe PaddleOCR GPU runtime"
set +e
"$PYTHON" "$repo_root/scripts/gpu_probe_paddle.py"
probe_rc=$?
set -e

if [[ "$probe_rc" -eq 0 ]]; then
  say "OK: PaddleOCR initialized with use_gpu=True"
  echo "This project: run server with --ocr-device cuda"
  echo "Example: maskpdf web --host 0.0.0.0 --port 8000 --ocr-device cuda"
  exit 0
fi

warn "PaddleOCR GPU probe failed (exit_code=$probe_rc)."
warn "Common causes:"
warn "- paddlepaddle is CPU-only build (paddle.is_compiled_with_cuda=false)"
warn "- CUDA/cuDNN mismatch for installed wheel"
warn "- NVIDIA driver missing or requires reboot"
warn "- running inside container without GPU access"

if [[ "$FIX" -eq 1 ]]; then
  warn "Fix mode already attempted installs. Next step is usually:"
  warn "- reboot if driver was installed/updated"
  warn "- install the correct paddlepaddle-gpu wheel for your CUDA version"
fi

exit "$probe_rc"


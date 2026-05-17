"""scripts/check_env.py — verify the environment before Phase 2+ workers run.

Three sanity checks the user calls out in memory/project_phase23_deps.md:

  1. PyTorch sees the GPU (skipped on CPU-only boxes; warning only).
  2. JAX sees the GPU (only when the [needle] extra is installed).
  3. flash-attn is importable (only when the [gpu] extra is installed; CUDA only).

Usage:  python scripts/check_env.py
"""

from __future__ import annotations

import sys


def _check_torch() -> int:
    try:
        import torch
    except ImportError as exc:
        print(f"[FAIL] torch import failed: {exc}")
        return 1
    if torch.cuda.is_available():
        print(f"[ ok ] torch CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("[warn] torch.cuda.is_available() is False (running on CPU)")
    return 0


def _check_jax() -> int:
    try:
        import jax
    except ImportError:
        print("[skip] jax not installed (install with `pip install -e \".[needle]\"`)")
        return 0
    devices = jax.devices()
    print(f"[ ok ] jax devices: {devices}")
    return 0


def _check_flash_attn() -> int:
    try:
        import flash_attn
    except ImportError:
        print(
            "[skip] flash-attn not installed "
            "(CUDA-only; install with `pip install flash-attn --no-build-isolation`)"
        )
        return 0
    print(f"[ ok ] flash-attn: {flash_attn.__version__}")
    return 0


def main() -> int:
    exit_code = 0
    exit_code |= _check_torch()
    exit_code |= _check_jax()
    exit_code |= _check_flash_attn()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

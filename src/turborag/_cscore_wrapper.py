"""ctypes wrapper for the C scoring kernel.

Compiles _cscore.c on first import and caches the shared library.
Falls back gracefully to the Python implementation if compilation fails.
"""
from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_lib: ctypes.CDLL | None = None
_load_attempted = False


def _find_or_compile() -> ctypes.CDLL | None:
    """Find pre-compiled library or compile from source."""
    src_dir = Path(__file__).parent
    c_source = src_dir / "_cscore.c"

    if not c_source.exists():
        return None

    # Try to find pre-compiled library next to the source
    if sys.platform == "darwin":
        lib_name = "_cscore.dylib"
    elif sys.platform == "win32":
        lib_name = "_cscore.dll"
    else:
        lib_name = "_cscore.so"

    lib_path = src_dir / lib_name
    source_mtime = c_source.stat().st_mtime

    # Recompile if source is newer than library
    if lib_path.exists() and lib_path.stat().st_mtime >= source_mtime:
        try:
            return ctypes.CDLL(str(lib_path))
        except OSError:
            pass

    # Compile
    try:
        logger.info("Compiling C scoring kernel: %s", c_source)
        cmd = [
            "gcc", "-O3", "-march=native", "-funroll-loops",
            "-shared", "-fPIC",
            "-o", str(lib_path),
            str(c_source),
        ]
        if sys.platform == "darwin":
            cmd.insert(1, "-undefined")
            cmd.insert(2, "dynamic_lookup")

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logger.warning("C kernel compilation failed: %s", result.stderr)
            return None

        return ctypes.CDLL(str(lib_path))
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Cannot compile C kernel (gcc not found?): %s", exc)
        return None


def _get_lib() -> ctypes.CDLL | None:
    """Get the compiled library, compiling on first access."""
    global _lib, _load_attempted
    if not _load_attempted:
        _load_attempted = True
        _lib = _find_or_compile()
        if _lib is not None:
            # Set up function signatures — original dispatch
            _lib.score_lut_dispatch.restype = ctypes.c_int
            _lib.score_lut_dispatch.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),   # packed_db
                ctypes.POINTER(ctypes.c_double),   # lut
                ctypes.POINTER(ctypes.c_double),   # scores_out
                ctypes.c_int,                       # n_vectors
                ctypes.c_int,                       # dim
                ctypes.c_int,                       # bits
                ctypes.c_int,                       # bytes_per_vec
            ]

            # Fused dispatch
            _lib.score_fused_dispatch.restype = ctypes.c_int
            _lib.score_fused_dispatch.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),   # packed_db
                ctypes.POINTER(ctypes.c_double),   # fused_lut
                ctypes.POINTER(ctypes.c_double),   # scores_out
                ctypes.c_int,                       # n_vectors
                ctypes.c_int,                       # n_bytes
                ctypes.c_int,                       # bytes_per_vec
            ]

            # Fused LUT builders
            _lib.build_fused_4bit.restype = None
            _lib.build_fused_4bit.argtypes = [
                ctypes.POINTER(ctypes.c_double),   # lut
                ctypes.POINTER(ctypes.c_double),   # fused out
                ctypes.c_int,                       # dim
            ]
            _lib.build_fused_2bit.restype = None
            _lib.build_fused_2bit.argtypes = [
                ctypes.POINTER(ctypes.c_double),   # lut
                ctypes.POINTER(ctypes.c_double),   # fused out
                ctypes.c_int,                       # dim
            ]

            logger.info("C scoring kernel loaded successfully")
        else:
            logger.info("Using Python scoring kernel (C extension not available)")
    return _lib


def score_lut_c(
    packed_db: NDArray[np.uint8],
    lut: NDArray[np.float64],
    dim: int,
    bits: int,
) -> NDArray[np.float32] | None:
    """Score using the C kernel. Returns None if C kernel is unavailable."""
    lib = _get_lib()
    if lib is None:
        return None

    packed = np.ascontiguousarray(packed_db, dtype=np.uint8)
    lut_c = np.ascontiguousarray(lut, dtype=np.float64)

    if packed.ndim == 1:
        packed = packed.reshape(1, -1)

    n_vectors = packed.shape[0]
    bytes_per_vec = packed.shape[1]

    scores = np.zeros(n_vectors, dtype=np.float64)

    ret = lib.score_lut_dispatch(
        packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        lut_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n_vectors),
        ctypes.c_int(dim),
        ctypes.c_int(bits),
        ctypes.c_int(bytes_per_vec),
    )

    if ret != 0:
        return None

    return scores.astype(np.float32)


def build_fused_lut_c(
    lut: NDArray[np.float64],
    dim: int,
    bits: int,
) -> NDArray[np.float64] | None:
    """Build a fused byte LUT in C. Returns None if unavailable."""
    lib = _get_lib()
    if lib is None:
        return None

    if bits == 4:
        n_bytes = (dim + 1) // 2
    elif bits == 2:
        n_bytes = (dim + 3) // 4
    else:
        return None  # 3-bit doesn't use fused LUT

    lut_c = np.ascontiguousarray(lut, dtype=np.float64)
    fused = np.zeros(n_bytes * 256, dtype=np.float64)

    if bits == 4:
        lib.build_fused_4bit(
            lut_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            fused.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(dim),
        )
    else:
        lib.build_fused_2bit(
            lut_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            fused.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(dim),
        )

    return fused


def score_fused_c(
    packed_db: NDArray[np.uint8],
    fused_lut: NDArray[np.float64],
    n_bytes: int,
) -> NDArray[np.float32] | None:
    """Score using a pre-built fused byte LUT. Returns None if unavailable."""
    lib = _get_lib()
    if lib is None:
        return None

    packed = np.ascontiguousarray(packed_db, dtype=np.uint8)
    fused_c = np.ascontiguousarray(fused_lut, dtype=np.float64)

    if packed.ndim == 1:
        packed = packed.reshape(1, -1)

    n_vectors = packed.shape[0]
    bytes_per_vec = packed.shape[1]

    scores = np.zeros(n_vectors, dtype=np.float64)

    ret = lib.score_fused_dispatch(
        packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        fused_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n_vectors),
        ctypes.c_int(n_bytes),
        ctypes.c_int(bytes_per_vec),
    )

    if ret != 0:
        return None

    return scores.astype(np.float32)


def is_available() -> bool:
    """Check if the C scoring kernel is available."""
    return _get_lib() is not None

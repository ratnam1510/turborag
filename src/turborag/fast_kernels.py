"""High-performance scoring kernels for compressed vector search.

Strategy: For each dimension, extract quantized levels for ALL vectors at once
(vectorized column extraction), then use the level as an index into a 1D LUT
row to get each vector's contribution. Accumulate into a running score array.

This approach:
- Avoids the massive (n_vectors, dim) float32 dequantized matrix (memory win)
- Uses vectorized NumPy column ops (no Python per-vector loop)
- Accumulates in float64 to avoid rounding on large shards
- Is cache-friendly: processes one LUT row at a time
"""
from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
Float64Array = NDArray[np.float64]
Uint8Array = NDArray[np.uint8]

SUPPORTED_BITS: Final[set[int]] = {2, 3, 4}


# ---------------------------------------------------------------------------
# LUT construction helpers
# ---------------------------------------------------------------------------

def build_query_lut(
    query_rotated: FloatArray,
    bits: int,
    value_range: float = 1.0,
) -> Float64Array:
    """Precompute a lookup table for a single rotated query vector.

    Returns a ``(dim, 2**bits)`` float64 table where ``lut[d][level]`` is the
    contribution to the dot product from dimension *d* when the database vector
    has quantized level *level*.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {sorted(SUPPORTED_BITS)}")

    q = np.asarray(query_rotated, dtype=np.float64).ravel()
    num_levels = 1 << bits
    max_level = num_levels - 1

    levels = np.arange(num_levels, dtype=np.float64)
    dequant_values = (levels / max_level) * (2.0 * value_range) - value_range

    return q[:, np.newaxis] * dequant_values[np.newaxis, :]  # (dim, num_levels)


# ---------------------------------------------------------------------------
# Scoring — single query
# ---------------------------------------------------------------------------

def score_shard_lut(
    packed_db: Uint8Array,
    lut: Float64Array,
    dim: int,
    bits: int,
) -> FloatArray:
    """Score all vectors in a packed shard using a precomputed LUT.

    *lut* must be a raw ``(dim, 2**bits)`` float64 array returned by
    :func:`build_query_lut`.

    Memory usage is O(n_vectors) regardless of dim — never allocates a full
    float32 dequantized matrix.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {sorted(SUPPORTED_BITS)}")

    packed = np.asarray(packed_db, dtype=np.uint8)
    if packed.ndim == 1:
        packed = packed.reshape(1, -1)

    # ---- Fallback: per-dimension C kernel ----
    try:
        from ._cscore_wrapper import score_lut_c

        c_result = score_lut_c(packed, lut, dim, bits)
        if c_result is not None:
            return c_result
    except ImportError:
        pass

    # ---- Pure-Python fallback ----
    n_vectors = packed.shape[0]
    scores = np.zeros(n_vectors, dtype=np.float64)

    if bits == 4:
        _accumulate_4bit(packed, lut, dim, scores)
    elif bits == 2:
        _accumulate_2bit(packed, lut, dim, scores)
    else:
        _accumulate_general(packed, lut, dim, bits, scores)

    return scores.astype(np.float32)


# ---------------------------------------------------------------------------
# Accumulation helpers (pure-Python fallback)
# ---------------------------------------------------------------------------

def _accumulate_4bit(
    packed: Uint8Array, lut: Float64Array, dim: int, scores: Float64Array
) -> None:
    """4-bit: 2 values per byte, no bit-spanning.

    Process pairs of dimensions from each byte for better throughput.
    """
    n_bytes = (dim + 1) // 2
    for byte_idx in range(n_bytes):
        d_even = byte_idx << 1
        column = packed[:, byte_idx]

        # Low nibble → even dimension
        if d_even < dim:
            level_even = column & 0x0F
            scores += lut[d_even][level_even]

        # High nibble → odd dimension
        d_odd = d_even + 1
        if d_odd < dim:
            level_odd = (column >> 4) & 0x0F
            scores += lut[d_odd][level_odd]


def _accumulate_2bit(
    packed: Uint8Array, lut: Float64Array, dim: int, scores: Float64Array
) -> None:
    """2-bit: 4 values per byte, no bit-spanning.

    Process 4 dimensions from each byte for maximum throughput.
    """
    n_bytes = (dim + 3) // 4
    for byte_idx in range(n_bytes):
        column = packed[:, byte_idx]
        base_d = byte_idx << 2

        for sub in range(4):
            d = base_d + sub
            if d >= dim:
                break
            level = (column >> (sub << 1)) & 0x03
            scores += lut[d][level]


def _accumulate_general(
    packed: Uint8Array, lut: Float64Array, dim: int, bits: int, scores: Float64Array
) -> None:
    """General path for 3-bit (handles byte-boundary spanning)."""
    mask = (1 << bits) - 1
    bit_offset = 0

    for d in range(dim):
        byte_idx = bit_offset >> 3
        shift = bit_offset & 7

        chunk = packed[:, byte_idx].astype(np.uint16) >> shift
        spill = shift + bits - 8
        if spill > 0:
            chunk |= packed[:, byte_idx + 1].astype(np.uint16) << (8 - shift)

        level = (chunk & mask).astype(np.intp)
        scores += lut[d][level]
        bit_offset += bits


# ---------------------------------------------------------------------------
# Scoring — batched queries
# ---------------------------------------------------------------------------

def score_shard_lut_batch(
    packed_db: Uint8Array,
    luts: list[Float64Array],
    dim: int,
    bits: int,
) -> FloatArray:
    """Score all vectors for multiple queries at once.

    Each element of *luts* must be a raw ``(dim, 2**bits)`` float64 array
    returned by :func:`build_query_lut`.

    Returns float32 array of shape ``(n_queries, n_vectors)``.
    """
    packed = np.asarray(packed_db, dtype=np.uint8)
    if packed.ndim == 1:
        packed = packed.reshape(1, -1)

    results = np.empty((len(luts), packed.shape[0]), dtype=np.float32)
    for qi, lut in enumerate(luts):
        results[qi] = score_shard_lut(packed, lut, dim, bits)
    return results

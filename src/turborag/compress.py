from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.stats import ortho_group

DEFAULT_VALUE_RANGE: Final[float] = 1.0
SUPPORTED_BITS: Final[set[int]] = {2, 3, 4}


FloatArray = NDArray[np.float32]
Uint8Array = NDArray[np.uint8]


def generate_rotation(dim: int, seed: int) -> FloatArray:
    """Generate a deterministic orthogonal rotation matrix."""
    if dim <= 0:
        raise ValueError("dim must be positive")

    if dim <= 2048:
        rotation = ortho_group.rvs(dim, random_state=int(seed)).astype(np.float32)
        return rotation

    block_size = 512
    blocks: list[FloatArray] = []
    offset = 0
    while offset < dim:
        current = min(block_size, dim - offset)
        blocks.append(ortho_group.rvs(current, random_state=int(seed) + offset).astype(np.float32))
        offset += current
    return linalg.block_diag(*blocks).astype(np.float32)


def normalize_rows(vectors: FloatArray) -> FloatArray:
    """L2-normalise a matrix of vectors."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.clip(norms, 1e-12, None)
    return (vectors / safe_norms).astype(np.float32)


def bytes_per_vector(dim: int, bits: int) -> int:
    _validate_bits(bits)
    if dim <= 0:
        raise ValueError("dim must be positive")
    return math.ceil(dim * bits / 8)


def quantize_qjl(
    x_rotated: FloatArray,
    bits: int = 3,
    value_range: float = DEFAULT_VALUE_RANGE,
) -> Uint8Array:
    """Uniformly quantize rotated vectors and bit-pack them.

    Despite the function name, this implementation uses scalar quantization rather than a
    full SRHT pipeline because the source spec presents an inconsistent definition of the
    quantization stage. The chosen representation is deterministic, stable across batches,
    and suitable for incremental indexing.
    """
    _validate_bits(bits)
    x = np.asarray(x_rotated, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError("x_rotated must be a 1D or 2D float32 array")
    if value_range <= 0:
        raise ValueError("value_range must be positive")

    levels = (1 << bits) - 1
    clipped = np.clip(x, -value_range, value_range)
    scaled = (clipped + value_range) / (2 * value_range)
    quantized = np.rint(scaled * levels).astype(np.uint8)
    return _pack_bits(quantized, bits)


def dequantize_qjl(
    x_packed: Uint8Array,
    dim: int,
    bits: int = 3,
    value_range: float = DEFAULT_VALUE_RANGE,
) -> FloatArray:
    """Unpack and dequantize vectors back into float32 space."""
    _validate_bits(bits)
    if value_range <= 0:
        raise ValueError("value_range must be positive")

    packed = np.asarray(x_packed, dtype=np.uint8)
    if packed.ndim == 1:
        packed = packed.reshape(1, -1)
    if packed.ndim != 2:
        raise ValueError("x_packed must be a 1D or 2D uint8 array")

    quantized = _unpack_bits(packed, dim=dim, bits=bits).astype(np.float32)
    levels = (1 << bits) - 1
    restored = (quantized / levels) * (2 * value_range) - value_range
    return restored.astype(np.float32)


def compressed_dot(
    q_packed: Uint8Array,
    db_packed: Uint8Array,
    dim: int,
    bits: int = 3,
    value_range: float = DEFAULT_VALUE_RANGE,
) -> FloatArray:
    """Approximate dot product in compressed space.

    This baseline implementation unpacks and dequantizes the vectors before using a dense
    matrix multiply. It is intentionally simple and should be replaced with a faster kernel
    once the numeric behavior is locked down.
    """
    query = dequantize_qjl(q_packed, dim=dim, bits=bits, value_range=value_range)
    database = dequantize_qjl(db_packed, dim=dim, bits=bits, value_range=value_range)
    if query.shape[0] != 1:
        raise ValueError("q_packed must represent exactly one query vector")
    return (database @ query[0]).astype(np.float32)


def _pack_bits(values: Uint8Array, bits: int) -> Uint8Array:
    _validate_bits(bits)
    array = np.asarray(values, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError("values must be a 2D uint8 array")

    rows, dim = array.shape
    packed = np.zeros((rows, bytes_per_vector(dim, bits)), dtype=np.uint8)
    mask = (1 << bits) - 1
    bit_offset = 0

    for column in range(dim):
        byte_index = bit_offset // 8
        shift = bit_offset % 8
        value = (array[:, column] & mask).astype(np.uint16)
        packed[:, byte_index] |= ((value << shift) & 0xFF).astype(np.uint8)
        spill = shift + bits - 8
        if spill > 0:
            packed[:, byte_index + 1] |= (value >> (8 - shift)).astype(np.uint8)
        bit_offset += bits

    return packed


def _unpack_bits(packed: Uint8Array, dim: int, bits: int) -> Uint8Array:
    _validate_bits(bits)
    array = np.asarray(packed, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError("packed must be a 2D uint8 array")

    rows = array.shape[0]
    unpacked = np.zeros((rows, dim), dtype=np.uint8)
    mask = (1 << bits) - 1
    bit_offset = 0

    for column in range(dim):
        byte_index = bit_offset // 8
        shift = bit_offset % 8
        chunk = (array[:, byte_index].astype(np.uint16) >> shift)
        spill = shift + bits - 8
        if spill > 0:
            chunk |= array[:, byte_index + 1].astype(np.uint16) << (8 - shift)
        unpacked[:, column] = (chunk & mask).astype(np.uint8)
        bit_offset += bits

    return unpacked


def _validate_bits(bits: int) -> None:
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {sorted(SUPPORTED_BITS)}")

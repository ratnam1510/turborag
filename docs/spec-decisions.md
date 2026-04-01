# Spec Decisions

This document records the places where the TurboQuant/QJL/PolarQuant papers left room for interpretation and the exact decisions used in this implementation.

Reference papers:
- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [Quantized Johnson-Lindenstrauss (QJL)](https://arxiv.org/abs/2406.03482) (AAAI 2025)
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026)

## 1. Quantization Strategy

### Spec Tension

The papers describe QJL, SRHT projection, sign quantization, and scalar min/max quantization with different tradeoffs.

### Decision

Implement rotated scalar quantization with fixed symmetric bounds.

### Why

- It is internally consistent.
- It supports incremental indexing without having to re-encode older shards.
- It keeps query-time and index-time calibration identical.
- It is easy to test and reason about.

## 2. Value Range

### Spec Tension

The pseudocode calibrates each batch with per-dimension min/max values, but query vectors need the same calibration to be comparable with stored vectors.

### Decision

Normalize vectors by default and quantize into a fixed range of `[-1.0, 1.0]`.

### Why

- Rotated unit vectors remain bounded.
- The representation is stable across batches.
- Incremental adds work naturally.

## 3. Rotation Persistence

### Spec Tension

The narrative suggests storing only a seed and algorithm identifier, while the storage layout explicitly includes `rotation.npy`.

### Decision

Store both the seed and the concrete `rotation.npy` matrix.

### Why

- Loading is deterministic and exact.
- The index file format is explicit.
- It avoids regeneration differences across SciPy versions.

## 4. Search Kernel

### Spec Tension

The PDF points toward a future SIMD or popcount-optimised kernel but also shows a dequantize-then-dot prototype.

### Decision

Start with the prototype dequantize-then-dot implementation.

### Why

- It is portable.
- It is readable.
- It establishes a correctness baseline before optimisation.

## 5. Graph Dependencies

### Spec Tension

The graph layer is central to the long-term product, but the spec also argues for a lightweight core install.

### Decision

Keep graph dependencies optional and degrade gracefully when they are not installed.

### Why

- The core package remains light.
- Dense retrieval remains useful on its own.
- The architecture still cleanly accommodates the graph layer.

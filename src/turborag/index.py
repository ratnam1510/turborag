from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .compress import (
    DEFAULT_VALUE_RANGE,
    bytes_per_vector,
    dequantize_qjl,
    generate_rotation,
    normalize_rows,
    quantize_qjl,
)
from .exceptions import DuplicateIDError, IDNotFoundError
from .fast_kernels import build_query_lut, build_query_lut_f32, build_query_weights_f32, score_shard_lut, topk_shard_lut

logger = logging.getLogger(__name__)


Uint8Array = NDArray[np.uint8]


@dataclass(slots=True)
class _Shard:
    ids: list[str]
    vectors: Uint8Array
    sketches: Uint8Array | None = None
    path: Path | None = None

    def __len__(self) -> int:
        return len(self.ids)


class TurboIndex:
    """Compressed vector index with optional shard persistence."""

    def __init__(
        self,
        dim: int,
        bits: int = 3,
        shard_size: int = 100_000,
        storage_dir: str | None = None,
        seed: int = 42,
        normalize: bool = True,
        value_range: float = DEFAULT_VALUE_RANGE,
        _skip_rotation: bool = False,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if value_range <= 0:
            raise ValueError("value_range must be positive")

        self.dim = dim
        self.bits = bits
        self.shard_size = shard_size
        self.seed = seed
        self.normalize = normalize
        self.value_range = value_range
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if _skip_rotation:
            self.rotation = np.eye(dim, dtype=np.float32)  # placeholder
            self._rotation_t = self.rotation
        else:
            self.rotation = generate_rotation(dim, seed)
            self._rotation_t = self.rotation.T.copy()
        self._bytes_per_vector = bytes_per_vector(dim, bits)
        self._shards: list[_Shard] = []
        self._ids: set[str] = set()
        self._size = 0
        self._sketch_bytes = (dim + 7) // 8  # ceil(dim/8) bytes for sign bits

        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            (self.storage_dir / "shards").mkdir(parents=True, exist_ok=True)

    def add(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(f"vectors must have shape (N, {self.dim})")
        if matrix.shape[0] != len(ids):
            raise ValueError("vectors and ids must have the same length")
        if len(set(ids)) != len(ids):
            raise ValueError("ids must be unique within a batch")
        duplicates = set(ids) & self._ids
        if duplicates:
            duplicate = sorted(duplicates)[0]
            raise DuplicateIDError(duplicate)

        offset = 0
        while offset < len(ids):
            end = min(offset + self.shard_size, len(ids))
            chunk = matrix[offset:end]
            working = normalize_rows(chunk) if self.normalize else chunk
            rotated = (working @ self.rotation.T).astype(np.float32)
            packed = quantize_qjl(rotated, bits=self.bits, value_range=self.value_range)
            # Generate binary sketches from rotated vectors (sign bits)
            sketch = np.packbits((rotated >= 0).astype(np.uint8), axis=1)
            # Trim to exact sketch_bytes (packbits may pad)
            sketch = sketch[:, :self._sketch_bytes]
            self._append_shard(packed, list(ids[offset:end]), sketch)
            offset = end

        self._ids.update(ids)
        self._size += len(ids)

    # Threshold: above this many vectors, use sketch pre-filter in auto mode
    _SKETCH_THRESHOLD = 10_000

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        *,
        mode: str = "auto",
    ) -> list[tuple[str, float]]:
        """Search the index for the k nearest vectors.

        Parameters
        ----------
        query : array of shape ``(dim,)`` or ``(1, dim)``
        k : number of results to return
        mode : search strategy
            ``"auto"`` — system picks the best strategy based on index size
            and sketch availability.
            ``"exact"`` — exhaustive LUT scan (guaranteed perfect recall).
            ``"fast"`` — binary sketch pre-filter + LUT refine
            (near-perfect recall, much faster on large indexes).
        """
        if k <= 0:
            return []
        if not self._shards:
            return []

        if mode == "auto":
            if self._has_sketches and self._size > self._SKETCH_THRESHOLD:
                mode = "fast"
            else:
                mode = "exact"

        if mode == "fast" and self._has_sketches:
            return self._search_sketch(query, k)
        return self._search_exact(query, k)

    def _search_exact(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Exhaustive scan — guaranteed perfect recall.

        Uses the weighted integer scorer for 3-bit (no LUT, direct arithmetic),
        falling back to LUT-based scoring for other bit widths or if the C
        kernel is unavailable.
        """
        query_rotated = self._prepare_query(query)

        # Try weighted integer scorer for 3-bit (faster, no LUT overhead)
        if self.bits == 3:
            result = self._search_exact_weighted(query_rotated, k)
            if result is not None:
                return result

        # Fallback: LUT-based exact scorer
        lut = self._build_exact_query_lut(query_rotated)

        # Single shard fast path
        if len(self._shards) == 1:
            shard = self._shards[0]
            top_indices, top_scores = topk_shard_lut(shard.vectors, lut, dim=self.dim, bits=self.bits, k=k)
            return [(shard.ids[int(i)], float(score)) for i, score in zip(top_indices, top_scores, strict=False)]

        # Multi-shard path
        candidates: list[tuple[str, float]] = []
        for shard in self._shards:
            shard_indices, shard_scores = topk_shard_lut(shard.vectors, lut, dim=self.dim, bits=self.bits, k=k)
            for index, score in zip(shard_indices, shard_scores, strict=False):
                candidates.append((shard.ids[int(index)], float(score)))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:k]

    def _search_exact_weighted(self, query_rotated: np.ndarray, k: int) -> list[tuple[str, float]] | None:
        """Weighted integer dot product scorer for 3-bit. Returns None if unavailable."""
        try:
            from ._cscore_wrapper import score_3bit_weighted_topk_c
        except ImportError:
            return None

        weights, bias = build_query_weights_f32(query_rotated, value_range=self.value_range)

        if len(self._shards) == 1:
            shard = self._shards[0]
            result = score_3bit_weighted_topk_c(shard.vectors, weights, bias, dim=self.dim, k=k)
            if result is None:
                return None
            top_indices, top_scores = result
            return [(shard.ids[int(i)], float(score)) for i, score in zip(top_indices, top_scores, strict=False)]

        candidates: list[tuple[str, float]] = []
        for shard in self._shards:
            result = score_3bit_weighted_topk_c(shard.vectors, weights, bias, dim=self.dim, k=k)
            if result is None:
                return None  # fall back to LUT for all shards
            shard_indices, shard_scores = result
            for index, score in zip(shard_indices, shard_scores, strict=False):
                candidates.append((shard.ids[int(index)], float(score)))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:k]

    def _search_sketch(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Two-stage search: binary sketch pre-filter + LUT refine."""
        query_rotated = self._prepare_query(query)

        # Build query sketch (sign bits of rotated query)
        query_sketch = np.packbits((query_rotated >= 0).astype(np.uint8))[:self._sketch_bytes]

        # Shortlist: 10% of index (tuned for high recall on varied data)
        shortlist_size = max(k * 20, self._size // 10)

        # --- Single-shard fast path (most common) ---
        if len(self._shards) == 1:
            shard = self._shards[0]
            if shard.sketches is None:
                return self._search_exact(query, k)
            distances = self._hamming_scan(shard.sketches, query_sketch)
            if distances is None:
                return self._search_exact(query, k)

            actual_n = min(shortlist_size, len(distances))
            if actual_n >= len(distances):
                candidate_idx = np.arange(len(distances))
            else:
                # Use partition to find threshold, then boolean mask (faster than argpartition)
                threshold = np.partition(distances, actual_n)[actual_n]
                mask = distances <= threshold
                candidate_idx = np.flatnonzero(mask)
                # If ties push us over, trim to shortlist_size
                if len(candidate_idx) > actual_n:
                    candidate_idx = candidate_idx[:actual_n]

            # Stage 2: full LUT on candidates
            lut = build_query_lut(query_rotated, bits=self.bits, value_range=self.value_range)
            candidate_vectors = np.ascontiguousarray(shard.vectors[candidate_idx], dtype=np.uint8)
            scores = score_shard_lut(candidate_vectors, lut, dim=self.dim, bits=self.bits)

            local_k = min(k, len(scores))
            if local_k == len(scores):
                top = np.argsort(scores)[::-1]
            else:
                top = np.argpartition(scores, -local_k)[-local_k:]
                top = top[np.argsort(scores[top])[::-1]]

            ids = shard.ids
            return [(ids[int(candidate_idx[i])], float(scores[int(i)])) for i in top]

        # --- Multi-shard path ---
        all_distances: list[tuple[int, np.ndarray]] = []
        for shard_idx, shard in enumerate(self._shards):
            if shard.sketches is None:
                return self._search_exact(query, k)
            distances = self._hamming_scan(shard.sketches, query_sketch)
            if distances is None:
                return self._search_exact(query, k)
            all_distances.append((shard_idx, distances))

        # Merge distances across shards
        total = sum(len(d) for _, d in all_distances)
        global_distances = np.empty(total, dtype=np.int32)
        global_shard_idx = np.empty(total, dtype=np.int32)
        global_local_idx = np.empty(total, dtype=np.int32)
        pos = 0
        for shard_idx, dists in all_distances:
            n = len(dists)
            global_distances[pos:pos + n] = dists
            global_shard_idx[pos:pos + n] = shard_idx
            global_local_idx[pos:pos + n] = np.arange(n)
            pos += n

        actual_shortlist = min(shortlist_size, total)
        if actual_shortlist >= total:
            candidate_indices = np.arange(total)
        else:
            candidate_indices = np.argpartition(global_distances, actual_shortlist)[:actual_shortlist]

        # Stage 2: full LUT on grouped candidates
        lut = build_query_lut(query_rotated, bits=self.bits, value_range=self.value_range)
        all_ids: list[str] = []
        all_scores: list[float] = []
        for shard_idx, shard in enumerate(self._shards):
            mask = global_shard_idx[candidate_indices] == shard_idx
            local_indices = global_local_idx[candidate_indices[mask]]
            if len(local_indices) == 0:
                continue
            candidate_vectors = np.ascontiguousarray(shard.vectors[local_indices], dtype=np.uint8)
            scores = score_shard_lut(candidate_vectors, lut, dim=self.dim, bits=self.bits)
            for i, li in enumerate(local_indices):
                all_ids.append(shard.ids[int(li)])
                all_scores.append(float(scores[i]))

        # Top-k from all scored candidates
        if len(all_scores) <= k:
            pairs = sorted(zip(all_ids, all_scores), key=lambda x: x[1], reverse=True)
            return [(cid, s) for cid, s in pairs]
        score_arr = np.array(all_scores, dtype=np.float32)
        top = np.argpartition(score_arr, -k)[-k:]
        top = top[np.argsort(score_arr[top])[::-1]]
        return [(all_ids[int(i)], all_scores[int(i)]) for i in top]

    @staticmethod
    def _hamming_scan(sketches: Uint8Array, query_sketch: np.ndarray) -> np.ndarray | None:
        """Run Hamming scan via C kernel or numpy fallback."""
        try:
            from ._cscore_wrapper import hamming_scan_c
            result = hamming_scan_c(sketches, query_sketch)
            if result is not None:
                return result
        except ImportError:
            pass
        # Numpy fallback using unpackbits
        xored = np.bitwise_xor(sketches, query_sketch[np.newaxis, :])
        return np.unpackbits(xored, axis=1).sum(axis=1).astype(np.int32)

    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 10,
        max_workers: int | None = None,
    ) -> list[list[tuple[str, float]]]:
        """Search for multiple queries at once.

        Parameters
        ----------
        queries : array of shape ``(n_queries, dim)``
        k : number of results per query
        max_workers : thread-pool size for parallel shard scanning.
            *None* uses a simple sequential loop.
        """
        matrix = np.asarray(queries, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(f"queries must have shape (N, {self.dim})")
        if k <= 0 or not self._shards:
            return [[] for _ in range(matrix.shape[0])]

        # Try batch weighted scorer for 3-bit (decodes each vector once)
        if self.bits == 3:
            result = self._search_batch_weighted(matrix, k)
            if result is not None:
                return result

        # Fallback: per-query LUT scoring
        prepared = [self._prepare_query(matrix[i]) for i in range(matrix.shape[0])]
        query_luts = [self._build_exact_query_lut(q) for q in prepared]

        all_candidates: list[list[tuple[str, float]]] = [[] for _ in range(len(query_luts))]

        def _scan_shard(shard: _Shard) -> list[list[tuple[str, float]]]:
            per_query: list[list[tuple[str, float]]] = []
            for qk in query_luts:
                idxs, top_scores = topk_shard_lut(shard.vectors, qk, dim=self.dim, bits=self.bits, k=k)
                if len(idxs) == 0:
                    per_query.append([])
                    continue
                per_query.append(
                    [(shard.ids[int(i)], float(score)) for i, score in zip(idxs, top_scores, strict=False)]
                )
            return per_query

        if max_workers is not None and max_workers > 1 and len(self._shards) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                shard_results = list(pool.map(_scan_shard, self._shards))
        else:
            shard_results = [_scan_shard(shard) for shard in self._shards]

        for shard_per_query in shard_results:
            for qi, hits in enumerate(shard_per_query):
                all_candidates[qi].extend(hits)

        results = []
        for candidates in all_candidates:
            candidates.sort(key=lambda item: item[1], reverse=True)
            results.append(candidates[:k])

        return results

    def _search_batch_weighted(
        self, queries: np.ndarray, k: int
    ) -> list[list[tuple[str, float]]] | None:
        """Batch scorer using vectorized query prep + native weighted batch C kernel."""
        try:
            from ._cscore_wrapper import score_3bit_weighted_batch_topk_c
        except ImportError:
            return None

        n_queries = queries.shape[0]

        # Vectorized query preparation (avoid per-query Python loop)
        matrix = np.asarray(queries, dtype=np.float32)
        if self.normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = (matrix / np.clip(norms, 1e-12, None)).astype(np.float32)
        rotated = (matrix @ self._rotation_t).astype(np.float32)

        # Build all weights + biases
        scale = np.float32(2.0 * self.value_range / 7.0)
        all_weights = np.ascontiguousarray(rotated * scale, dtype=np.float32)
        all_biases = np.ascontiguousarray(-self.value_range * rotated.sum(axis=1), dtype=np.float32)

        if len(self._shards) == 1:
            shard = self._shards[0]
            shard_results = score_3bit_weighted_batch_topk_c(
                shard.vectors,
                all_weights,
                all_biases,
                dim=self.dim,
                k=k,
            )
            if shard_results is None:
                return None
            return [
                [(shard.ids[int(idx)], float(score)) for idx, score in zip(indices, scores, strict=False)]
                for indices, scores in shard_results
            ]

        all_candidates: list[list[tuple[str, float]]] = [[] for _ in range(n_queries)]
        for shard in self._shards:
            shard_results = score_3bit_weighted_batch_topk_c(
                shard.vectors,
                all_weights,
                all_biases,
                dim=self.dim,
                k=k,
            )
            if shard_results is None:
                return None
            for qi, (indices, scores) in enumerate(shard_results):
                all_candidates[qi].extend(
                    (shard.ids[int(idx)], float(score))
                    for idx, score in zip(indices, scores, strict=False)
                )

        return [sorted(candidates, key=lambda item: item[1], reverse=True)[:k] for candidates in all_candidates]

    def _prepare_query(self, query: np.ndarray) -> np.ndarray:
        """Validate, normalize, and rotate a single query vector. Returns float32."""
        vector = np.asarray(query, dtype=np.float32)
        if vector.ndim == 2:
            if vector.shape != (1, self.dim):
                raise ValueError(f"query matrix must have shape (1, {self.dim})")
            vector = vector[0]
        if vector.ndim != 1 or vector.shape[0] != self.dim:
            raise ValueError(f"query must have shape ({self.dim},)")

        if self.normalize:
            norm = np.linalg.norm(vector)
            vector = (vector / max(norm, 1e-12)).astype(np.float32)

        rotated = (vector.reshape(1, -1) @ self._rotation_t).astype(np.float32)
        return rotated[0]

    def _build_exact_query_lut(self, query_rotated: np.ndarray) -> np.ndarray:
        """Use float32 LUTs when exact mode can stay on the native 3-bit path."""
        if self.bits == 3:
            return build_query_lut_f32(query_rotated, bits=self.bits, value_range=self.value_range)
        return build_query_lut(query_rotated, bits=self.bits, value_range=self.value_range)

    def save(self, path: str) -> None:
        index_dir = Path(path)
        index_dir.mkdir(parents=True, exist_ok=True)
        shards_dir = index_dir / "shards"
        if shards_dir.exists():
            shutil.rmtree(shards_dir)
        shards_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "dim": self.dim,
            "bits": self.bits,
            "seed": self.seed,
            "shard_size": self.shard_size,
            "normalize": self.normalize,
            "value_range": self.value_range,
            "schema_version": 1,
            "size": self._size,
        }
        (index_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        np.save(index_dir / "rotation.npy", self.rotation)

        for shard_index, shard in enumerate(self._shards):
            shard_path = shards_dir / f"shard_{shard_index:03d}.bin"
            ids_path = shards_dir / f"shard_{shard_index:03d}.ids.json"
            np.asarray(shard.vectors, dtype=np.uint8).tofile(shard_path)
            ids_path.write_text(json.dumps(shard.ids, indent=2), encoding="utf-8")
            if shard.sketches is not None:
                sketch_path = shards_dir / f"shard_{shard_index:03d}.sketch.bin"
                np.asarray(shard.sketches, dtype=np.uint8).tofile(sketch_path)

    def load(self, path: str) -> None:
        index_dir = Path(path)
        config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))

        self.dim = int(config["dim"])
        self.bits = int(config["bits"])
        self.seed = int(config["seed"])
        self.shard_size = int(config["shard_size"])
        self.normalize = bool(config.get("normalize", True))
        self.value_range = float(config.get("value_range", DEFAULT_VALUE_RANGE))
        self.storage_dir = index_dir
        self.rotation = np.load(index_dir / "rotation.npy").astype(np.float32)
        self._rotation_t = self.rotation.T.copy()
        self._bytes_per_vector = bytes_per_vector(self.dim, self.bits)
        self._sketch_bytes = (self.dim + 7) // 8
        self._shards = []
        self._ids = set()
        self._size = 0

        shards_dir = index_dir / "shards"
        for ids_path in sorted(shards_dir.glob("*.ids.json")):
            shard_stem = ids_path.name.replace(".ids.json", "")
            shard_path = shards_dir / f"{shard_stem}.bin"
            ids = json.loads(ids_path.read_text(encoding="utf-8"))
            shape = (len(ids), self._bytes_per_vector)
            vectors = np.memmap(shard_path, dtype=np.uint8, mode="r", shape=shape)
            sketch_path = shards_dir / f"{shard_stem}.sketch.bin"
            shard_sketches = None
            if sketch_path.exists():
                sketch_shape = (len(ids), self._sketch_bytes)
                shard_sketches = np.memmap(sketch_path, dtype=np.uint8, mode="r", shape=sketch_shape)
            self._shards.append(_Shard(ids=ids, vectors=vectors, sketches=shard_sketches, path=shard_path))
            self._ids.update(ids)
            self._size += len(ids)

    @classmethod
    def open(cls, path: str) -> "TurboIndex":
        config_path = Path(path) / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        index = cls(
            dim=int(config["dim"]),
            bits=int(config["bits"]),
            shard_size=int(config["shard_size"]),
            storage_dir=path,
            seed=int(config["seed"]),
            normalize=bool(config.get("normalize", True)),
            value_range=float(config.get("value_range", DEFAULT_VALUE_RANGE)),
            _skip_rotation=True,
        )
        index.load(path)
        return index

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (
            f"TurboIndex(dim={self.dim}, bits={self.bits}, size={self._size}, "
            f"shards={len(self._shards)}, normalize={self.normalize})"
        )

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._ids

    @property
    def _has_sketches(self) -> bool:
        """Check if all shards have binary sketches available."""
        return bool(self._shards) and all(s.sketches is not None for s in self._shards)

    def delete(self, ids: Sequence[str]) -> int:
        """Remove vectors by ID. Returns the number of vectors actually removed.

        This rebuilds affected shards in-memory. For persisted indexes, call
        :meth:`save` afterward to write the updated shards to disk.
        """
        to_delete = set(ids) & self._ids
        if not to_delete:
            return 0

        new_shards: list[_Shard] = []
        for shard in self._shards:
            keep_mask = [i for i, sid in enumerate(shard.ids) if sid not in to_delete]
            if not keep_mask:
                continue  # entire shard deleted
            if len(keep_mask) == len(shard.ids):
                new_shards.append(shard)  # no deletions in this shard
            else:
                new_ids = [shard.ids[i] for i in keep_mask]
                new_vectors = np.asarray(shard.vectors, dtype=np.uint8)[keep_mask]
                new_sketches = np.asarray(shard.sketches, dtype=np.uint8)[keep_mask].copy() if shard.sketches is not None else None
                new_shards.append(_Shard(ids=new_ids, vectors=new_vectors.copy(), sketches=new_sketches))

        self._shards = new_shards
        self._ids -= to_delete
        removed = len(to_delete)
        self._size -= removed
        logger.debug("Deleted %d vectors from index", removed)
        return removed

    def update(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        """Update existing vectors by ID. Equivalent to delete + add.

        All IDs must already exist in the index.

        Parameters
        ----------
        vectors : array of shape ``(N, dim)``
        ids : sequence of N chunk IDs to update
        """
        for chunk_id in ids:
            if chunk_id not in self._ids:
                raise IDNotFoundError(chunk_id)

        self.delete(ids)
        self.add(vectors, ids)

    def _append_shard(self, packed_vectors: Uint8Array, ids: list[str], sketches: Uint8Array | None = None) -> None:
        if self.storage_dir is None:
            self._shards.append(_Shard(ids=ids, vectors=np.array(packed_vectors, copy=True), sketches=np.array(sketches, copy=True) if sketches is not None else None))
            return

        shard_index = len(self._shards)
        shard_path = self.storage_dir / "shards" / f"shard_{shard_index:03d}.bin"
        ids_path = self.storage_dir / "shards" / f"shard_{shard_index:03d}.ids.json"
        packed_vectors = np.asarray(packed_vectors, dtype=np.uint8)
        packed_vectors.tofile(shard_path)
        ids_path.write_text(json.dumps(ids, indent=2), encoding="utf-8")
        memmap = np.memmap(shard_path, dtype=np.uint8, mode="r", shape=packed_vectors.shape)

        shard_sketches = None
        if sketches is not None:
            sketch_path = self.storage_dir / "shards" / f"shard_{shard_index:03d}.sketch.bin"
            sketches_arr = np.asarray(sketches, dtype=np.uint8)
            sketches_arr.tofile(sketch_path)
            shard_sketches = np.memmap(sketch_path, dtype=np.uint8, mode="r", shape=sketches_arr.shape)

        self._shards.append(_Shard(ids=ids, vectors=memmap, sketches=shard_sketches, path=shard_path))

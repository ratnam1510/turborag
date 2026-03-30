from __future__ import annotations

import json
import logging
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
from .fast_kernels import build_query_lut, score_shard_lut

logger = logging.getLogger(__name__)


Uint8Array = NDArray[np.uint8]


@dataclass(slots=True)
class _Shard:
    ids: list[str]
    vectors: Uint8Array
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
        self.rotation = generate_rotation(dim, seed)
        self._rotation_t = self.rotation.T.copy()  # cached transpose for hot path
        self._bytes_per_vector = bytes_per_vector(dim, bits)
        self._shards: list[_Shard] = []
        self._ids: set[str] = set()
        self._size = 0

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

        working = normalize_rows(matrix) if self.normalize else matrix
        rotated = (working @ self.rotation.T).astype(np.float32)
        packed = quantize_qjl(rotated, bits=self.bits, value_range=self.value_range)

        offset = 0
        while offset < len(ids):
            next_offset = min(offset + self.shard_size, len(ids))
            batch_ids = list(ids[offset:next_offset])
            batch_vectors = packed[offset:next_offset]
            self._append_shard(batch_vectors, batch_ids)
            offset = next_offset

        self._ids.update(ids)
        self._size += len(ids)

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        if not self._shards:
            return []

        query_rotated = self._prepare_query(query)
        lut = build_query_lut(query_rotated, bits=self.bits, value_range=self.value_range)

        # Single shard fast path — skip candidate merging overhead
        if len(self._shards) == 1:
            shard = self._shards[0]
            scores = score_shard_lut(shard.vectors, lut, dim=self.dim, bits=self.bits)
            local_k = min(k, len(scores))
            if local_k == 0:
                return []
            if local_k == len(scores):
                top_indices = np.argsort(scores)[::-1]
            else:
                top_indices = np.argpartition(scores, -local_k)[-local_k:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            return [(shard.ids[int(i)], float(scores[int(i)])) for i in top_indices]

        # Multi-shard path
        candidates: list[tuple[str, float]] = []
        for shard in self._shards:
            scores = score_shard_lut(shard.vectors, lut, dim=self.dim, bits=self.bits)
            local_k = min(k, len(scores))
            if local_k == 0:
                continue
            if local_k == len(scores):
                shard_indices = np.arange(len(scores))
            else:
                shard_indices = np.argpartition(scores, -local_k)[-local_k:]
            for index in shard_indices:
                candidates.append((shard.ids[int(index)], float(scores[int(index)])))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:k]

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

        # Prepare all queries
        prepared = [self._prepare_query(matrix[i]) for i in range(matrix.shape[0])]
        luts = [build_query_lut(q, bits=self.bits, value_range=self.value_range) for q in prepared]

        all_candidates: list[list[tuple[str, float]]] = [[] for _ in range(len(luts))]

        def _scan_shard(shard: _Shard) -> list[list[tuple[str, float]]]:
            per_query: list[list[tuple[str, float]]] = []
            for lut in luts:
                scores = score_shard_lut(shard.vectors, lut, dim=self.dim, bits=self.bits)
                local_k = min(k, len(scores))
                if local_k == 0:
                    per_query.append([])
                    continue
                if local_k == len(scores):
                    idxs = np.arange(len(scores))
                else:
                    idxs = np.argpartition(scores, -local_k)[-local_k:]
                per_query.append([(shard.ids[int(i)], float(scores[int(i)])) for i in idxs])
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

    def save(self, path: str) -> None:
        index_dir = Path(path)
        index_dir.mkdir(parents=True, exist_ok=True)
        shards_dir = index_dir / "shards"
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
            self._shards.append(_Shard(ids=ids, vectors=vectors, path=shard_path))
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
                new_shards.append(_Shard(ids=new_ids, vectors=new_vectors.copy()))

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

    def _append_shard(self, packed_vectors: Uint8Array, ids: list[str]) -> None:
        if self.storage_dir is None:
            self._shards.append(_Shard(ids=ids, vectors=np.array(packed_vectors, copy=True)))
            return

        shard_index = len(self._shards)
        shard_path = self.storage_dir / "shards" / f"shard_{shard_index:03d}.bin"
        ids_path = self.storage_dir / "shards" / f"shard_{shard_index:03d}.ids.json"
        packed_vectors = np.asarray(packed_vectors, dtype=np.uint8)
        packed_vectors.tofile(shard_path)
        ids_path.write_text(json.dumps(ids, indent=2), encoding="utf-8")
        memmap = np.memmap(shard_path, dtype=np.uint8, mode="r", shape=packed_vectors.shape)
        self._shards.append(_Shard(ids=ids, vectors=memmap, path=shard_path))

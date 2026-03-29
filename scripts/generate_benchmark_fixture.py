#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic local TurboRAG benchmark fixture.")
    parser.add_argument("--output-dir", required=True, help="Directory where corpus.jsonl and queries.jsonl will be written.")
    parser.add_argument("--num-docs", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_docs <= 0 or args.dim <= 0 or args.num_queries <= 0:
        raise SystemExit("num-docs, dim, and num-queries must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    corpus = rng.normal(size=(args.num_docs, args.dim)).astype(np.float32)
    corpus /= np.maximum(np.linalg.norm(corpus, axis=1, keepdims=True), 1e-12)
    ids = [f"doc-{index}" for index in range(args.num_docs)]

    corpus_path = output_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for index, (chunk_id, vector) in enumerate(zip(ids, corpus, strict=False)):
            payload = {
                "chunk_id": chunk_id,
                "text": f"Synthetic benchmark document {index}",
                "embedding": vector.astype(float).tolist(),
                "metadata": {"fixture": "synthetic", "doc_index": index},
            }
            handle.write(json.dumps(payload) + "\n")

    queries_path = output_dir / "queries.jsonl"
    query_indices = rng.choice(args.num_docs, size=min(args.num_queries, args.num_docs), replace=False)
    with queries_path.open("w", encoding="utf-8") as handle:
        for query_index, doc_index in enumerate(query_indices):
            vector = corpus[int(doc_index)] + args.noise * rng.normal(size=args.dim).astype(np.float32)
            payload = {
                "query_id": f"q-{query_index}",
                "query_vector": vector.astype(float).tolist(),
                "relevant_ids": [ids[int(doc_index)]],
            }
            handle.write(json.dumps(payload) + "\n")

    manifest = {
        "seed": args.seed,
        "num_docs": args.num_docs,
        "dim": args.dim,
        "num_queries": min(args.num_queries, args.num_docs),
        "noise": args.noise,
        "corpus_path": str(corpus_path),
        "queries_path": str(queries_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

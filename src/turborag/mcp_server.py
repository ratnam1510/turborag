"""Lightweight MCP (Model Context Protocol) server for TurboRAG over stdio.

This module exposes TurboRAG retrieval as an MCP tool server that communicates
over stdin/stdout using the ``mcp`` Python library. It requires no hosting—just
run it as a subprocess from any MCP-capable client (e.g. Claude Desktop).

Usage::

    turborag mcp --index ./my_index

Or directly::

    python -m turborag.mcp_server --index ./my_index

The server exposes one tool:

- **turborag_query**: Search a TurboRAG sidecar index by a pre-computed
  query vector and return ranked results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _build_server(index_path: str):
    """Construct and return a configured MCP Server instance."""
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

    from .index import TurboIndex
    from .ingest import load_records_snapshot, SNAPSHOT_FILE_NAME

    import numpy as np

    index_dir = Path(index_path)
    index = TurboIndex.open(str(index_dir))

    snapshot_path = index_dir / SNAPSHOT_FILE_NAME
    records = load_records_snapshot(snapshot_path) if snapshot_path.exists() else {}

    server = Server("turborag")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="turborag_query",
                description=(
                    "Search the TurboRAG index with a query vector and return "
                    "the top-k most similar chunks with scores."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query_vector": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "The query embedding vector as a list of floats.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default 5).",
                            "default": 5,
                        },
                    },
                    "required": ["query_vector"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name != "turborag_query":
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        raw_vector = arguments.get("query_vector")
        if raw_vector is None:
            return [TextContent(type="text", text="Error: query_vector is required")]

        top_k = int(arguments.get("top_k", 5))
        vector = np.asarray(raw_vector, dtype=np.float32)

        hits = index.search(vector, k=top_k)
        results = []
        for chunk_id, score in hits:
            entry = {"chunk_id": chunk_id, "score": round(float(score), 6)}
            rec = records.get(chunk_id)
            if rec is not None:
                entry["text"] = rec.text
                entry["source_doc"] = rec.source_doc
                entry["page_num"] = rec.page_num
            results.append(entry)

        return [TextContent(type="text", text=json.dumps(results, indent=2))]

    return server, stdio_server


async def run_server(index_path: str) -> None:
    """Start the MCP server over stdio."""
    server, stdio_server_ctx = _build_server(index_path)

    async with stdio_server_ctx() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """CLI entrypoint: ``python -m turborag.mcp_server --index <path>``."""
    import argparse

    parser = argparse.ArgumentParser(description="TurboRAG MCP server (stdio)")
    parser.add_argument("--index", required=True, help="Path to a TurboRAG sidecar index directory")
    args = parser.parse_args()

    import asyncio
    asyncio.run(run_server(args.index))


if __name__ == "__main__":
    main()

"""Metadata filter evaluation for filtered vector search.

Filter specs use MongoDB-style operators::

    {"field": value}                  # equality (implicit $eq)
    {"field": {"$gt": value}}         # greater than
    {"field": {"$gte": value}}        # greater than or equal
    {"field": {"$lt": value}}         # less than
    {"field": {"$lte": value}}        # less than or equal
    {"field": {"$ne": value}}         # not equal
    {"field": {"$in": [v1, v2]}}      # in set
    {"field": {"$nin": [v1, v2]}}     # not in set
    {"field": {"$exists": true}}      # field exists

Multiple fields are ANDed implicitly::

    {"category": "finance", "year": {"$gte": 2024}}
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

_OPERATORS = frozenset({"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$exists"})


def match(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Return True if *metadata* satisfies every condition in *filters*."""
    for field, condition in filters.items():
        if not _match_field(metadata, field, condition):
            return False
    return True


def match_mask(
    metadata_list: list[dict[str, Any] | None],
    filters: dict[str, Any],
) -> NDArray[np.bool_]:
    """Return a boolean mask over *metadata_list* for entries matching *filters*."""
    n = len(metadata_list)
    mask = np.empty(n, dtype=np.bool_)
    for i in range(n):
        meta = metadata_list[i]
        mask[i] = meta is not None and match(meta, filters)
    return mask


def validate_filters(filters: dict[str, Any]) -> None:
    """Raise ``ValueError`` if *filters* contains unknown operators."""
    for condition in filters.values():
        if isinstance(condition, dict):
            unknown = set(condition) - _OPERATORS
            if unknown:
                raise ValueError(f"Unknown filter operators: {', '.join(sorted(unknown))}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _match_field(metadata: dict[str, Any], field: str, condition: Any) -> bool:
    if isinstance(condition, dict):
        for op, value in condition.items():
            if op not in _OPERATORS:
                raise ValueError(f"Unknown filter operator: {op}")
            if not _eval_op(metadata, field, op, value):
                return False
        return True
    return _eval_op(metadata, field, "$eq", condition)


def _eval_op(metadata: dict[str, Any], field: str, op: str, value: Any) -> bool:
    if op == "$exists":
        return (field in metadata) == bool(value)

    if field not in metadata:
        return False

    actual = metadata[field]

    if op == "$eq":
        return actual == value
    if op == "$ne":
        return actual != value
    if op == "$in":
        return actual in value
    if op == "$nin":
        return actual not in value

    # Comparison operators — guard against incomparable types
    try:
        if op == "$gt":
            return actual > value
        if op == "$gte":
            return actual >= value
        if op == "$lt":
            return actual < value
        if op == "$lte":
            return actual <= value
    except TypeError:
        return False

    return False

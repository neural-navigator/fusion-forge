from __future__ import annotations

import time
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from . import config


def _ids_from_points(points) -> list[str]:
    out = []
    for p in points:
        payload = p.payload or {}
        out.append(str(payload.get("doc_id", p.id)))
    return out


def search_dense(
    c: QdrantClient,
    collection: str,
    vector: list[float],
    limit: int = config.QUERY_K,
    using: str | None = None,
    search_params: models.SearchParams | None = None,
) -> tuple[list[str], list[float], float]:
    t0 = time.perf_counter()
    res = c.query_points(
        collection_name=collection,
        query=vector,
        using=using,
        limit=limit + 1,
        with_payload=True,
        search_params=search_params,
    )
    dt = time.perf_counter() - t0
    pts = res.points
    return _ids_from_points(pts), [float(p.score) for p in pts], dt


def search_sparse(
    c: QdrantClient,
    collection: str,
    sparse: dict,
    limit: int = config.QUERY_K,
    using: str = "bm42",
) -> tuple[list[str], list[float], float]:
    t0 = time.perf_counter()
    res = c.query_points(
        collection_name=collection,
        query=models.SparseVector(indices=sparse["indices"], values=sparse["values"]),
        using=using,
        limit=limit + 1,
        with_payload=True,
    )
    dt = time.perf_counter() - t0
    pts = res.points
    return _ids_from_points(pts), [float(p.score) for p in pts], dt


def search_fusion(
    c: QdrantClient,
    collection: str,
    dense_vec: list[float],
    sparse: dict,
    fusion: models.Fusion,
    limit: int = config.QUERY_K,
    prefetch: int = config.PREFETCH_LIMIT,
    dense_using: str = "dense",
    sparse_using: str = "bm42",
) -> tuple[list[str], list[float], float]:
    t0 = time.perf_counter()
    res = c.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(query=dense_vec, using=dense_using, limit=prefetch),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse["indices"], values=sparse["values"]
                ),
                using=sparse_using,
                limit=prefetch,
            ),
        ],
        query=models.FusionQuery(fusion=fusion),
        limit=limit + 1,
        with_payload=True,
    )
    dt = time.perf_counter() - t0
    pts = res.points
    return _ids_from_points(pts), [float(p.score) for p in pts], dt


def rrf_merge(
    rankings: list[list[str]], k: int = 60, limit: int = config.QUERY_K
) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, did in enumerate(ranking, start=1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]]


def weighted_merge(
    dense_ids: list[str],
    dense_scores: list[float],
    sparse_ids: list[str],
    sparse_scores: list[float],
    dense_w: float,
    sparse_w: float,
    limit: int = config.QUERY_K,
) -> list[str]:
    def norm(ids: list[str], scores: list[float]) -> dict[str, float]:
        if not scores:
            return {}
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        return {i: (s - lo) / span for i, s in zip(ids, scores)}

    nd, ns = norm(dense_ids, dense_scores), norm(sparse_ids, sparse_scores)
    keys = set(nd) | set(ns)
    fused = {k: dense_w * nd.get(k, 0.0) + sparse_w * ns.get(k, 0.0) for k in keys}
    return [d for d, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:limit]]


def payloads_for_ids(
    c: QdrantClient, collection: str, doc_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Fetch payloads by scrolling with a filter on doc_id (small corpora)."""
    wanted = set(doc_ids)
    found: dict[str, dict[str, Any]] = {}
    offset = None
    while True:
        pts, offset = c.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in pts:
            did = str((p.payload or {}).get("doc_id", p.id))
            if did in wanted:
                found[did] = p.payload or {}
        if offset is None or len(found) == len(wanted):
            break
    return found

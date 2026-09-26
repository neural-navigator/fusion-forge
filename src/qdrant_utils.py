from __future__ import annotations

import json
import time
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from . import config


def client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, timeout=300, check_compatibility=False)


def wait_ready(c: QdrantClient | None = None, retries: int = 20) -> QdrantClient:
    c = c or client()
    for _ in range(retries):
        try:
            c.get_collections()
            return c
        except Exception:
            time.sleep(1)
    raise RuntimeError("Qdrant is not reachable at " + config.QDRANT_URL)


def recreate_dense(
    c: QdrantClient,
    name: str,
    distance: models.Distance,
    quantization: models.QuantizationConfig | None = None,
    on_disk: bool = False,
) -> None:
    if c.collection_exists(name):
        c.delete_collection(name)
    c.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=config.DENSE_DIM,
            distance=distance,
            on_disk=on_disk,
        ),
        quantization_config=quantization,
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=200),
    )


def recreate_sparse(c: QdrantClient, name: str, use_idf: bool, on_disk: bool = True) -> None:
    if c.collection_exists(name):
        c.delete_collection(name)
    params = models.SparseVectorParams(index=models.SparseIndexParams(on_disk=on_disk))
    if use_idf:
        params = models.SparseVectorParams(
            modifier=models.Modifier.IDF,
            index=models.SparseIndexParams(on_disk=on_disk),
        )
    c.create_collection(
        collection_name=name,
        vectors_config={},
        sparse_vectors_config={"bm42": params},
    )


def recreate_hybrid(c: QdrantClient, name: str, on_disk: bool = True) -> None:
    if c.collection_exists(name):
        c.delete_collection(name)
    c.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(
                size=config.DENSE_DIM,
                distance=models.Distance.COSINE,
                on_disk=on_disk,
            )
        },
        sparse_vectors_config={
            "bm42": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
                index=models.SparseIndexParams(on_disk=on_disk),
            )
        },
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=20000,
            default_segment_number=1,
        ),
        wal_config=models.WalConfigDiff(wal_capacity_mb=8),
        on_disk_payload=True,
    )


def drop_extra_scifact_collections(c: QdrantClient) -> list[str]:
    extra = [
        config.COL_DENSE_DOT,
        config.COL_DENSE_EUCLID,
        config.COL_BM42_NO_IDF,
        config.COL_Q_SCALAR,
        config.COL_Q_BINARY,
        config.COL_Q_PQ,
        config.COL_DENSE,
        config.COL_BM42,
        "scifact_hybrid",
    ]
    dropped = []
    for name in extra:
        if c.collection_exists(name):
            c.delete_collection(name)
            dropped.append(name)
    return dropped


def upsert_points(c: QdrantClient, name: str, points: list[models.PointStruct]) -> None:
    batch = max(int(config.UPSERT_BATCH), 1)
    for start in range(0, len(points), batch):
        chunk = points[start : start + batch]
        last_err = None
        for attempt in range(5):
            try:
                c.upsert(collection_name=name, points=chunk, wait=False)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                time.sleep(2 + attempt * 2)
                c = wait_ready()
        if last_err is not None:
            raise last_err


def collection_resources(c: QdrantClient, name: str) -> dict[str, Any]:
    info = c.get_collection(name)
    raw = json.loads(info.model_dump_json())
    storage = config.ROOT / "qdrant_storage" / "collections" / name
    disk_bytes = 0
    if storage.exists():
        try:
            import subprocess

            # -sk is allocated 1KiB blocks. -sb is apparent size and inflates sparse WAL/payload pages.
            out = subprocess.check_output(["du", "-sk", str(storage)], text=True)
            disk_bytes = int(out.split()[0]) * 1024
        except Exception:
            disk_bytes = 0
            for p in storage.rglob("*"):
                if p.is_file():
                    disk_bytes += p.stat().st_blocks * 512
    return {
        "points_count": info.points_count,
        "indexed_vectors_count": info.indexed_vectors_count,
        "status": str(info.status),
        "segments_count": getattr(info, "segments_count", None),
        "disk_bytes": disk_bytes,
        "raw": {
            k: raw.get(k)
            for k in ("points_count", "indexed_vectors_count", "status", "optimizer_status")
        },
        "disk_mb": round(disk_bytes / (1024 * 1024), 3),
    }

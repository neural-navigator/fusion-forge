from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from qdrant_client.http import models
from tqdm import tqdm

from . import config, data, embeddings, qdrant_utils


def _free_gb() -> float:
    usage = shutil.disk_usage("/")
    return usage.free / (1024**3)


def index_hybrid_stream(
    dataset: str,
    zip_path,
    corpus_member: str,
    n_docs: int,
    store_text: bool,
) -> dict[str, Any]:
    c = qdrant_utils.wait_ready()
    name = config.hybrid_collection(dataset)
    t0 = time.perf_counter()
    qdrant_utils.recreate_hybrid(c, name, on_disk=True)
    pid = 0
    batch_rows: list[dict] = []
    n_ok = 0
    if Path(zip_path).stat().st_size > 400_000_000:
        dest = config.DATA_DIR / "extract" / corpus_member.replace("/", "_")
        data.extract_zip_member(Path(zip_path), corpus_member, dest)
        iterator = data.iter_jsonl_file(dest)
    else:
        iterator = data.iter_jsonl_from_zip(zip_path, corpus_member)
    pbar = tqdm(iterator, total=n_docs, desc=f"index {dataset}", unit="doc")
    for row in pbar:
        batch_rows.append(row)
        if len(batch_rows) >= config.UPSERT_BATCH:
            n_ok += _flush(c, name, batch_rows, pid, store_text)
            pid += len(batch_rows)
            batch_rows = []
    if batch_rows:
        n_ok += _flush(c, name, batch_rows, pid, store_text)
        pid += len(batch_rows)
    dt = time.perf_counter() - t0
    resources = qdrant_utils.collection_resources(c, name)
    print(f"Indexed {name}: {dt:.1f}s, {resources.get('points_count')} pts, {resources.get('disk_mb')} MB")
    return {
        "collection": name,
        "seconds": round(dt, 3),
        "upserted": pid,
        "resources": resources,
    }


def _flush(c, name, rows, start_id, store_text) -> int:
    texts = [data.doc_text(r) for r in rows]
    dense = embeddings.embed_dense(texts, query=False)
    sparse = embeddings.embed_sparse(texts)
    points = []
    for j, row in enumerate(rows):
        payload = {"doc_id": str(row["_id"])}
        if store_text:
            payload["title"] = (row.get("title") or "")[:500]
            payload["text"] = (row.get("text") or "")[:2000]
        sv = sparse[j]
        points.append(
            models.PointStruct(
                id=start_id + j,
                vector={
                    "dense": dense[j].tolist(),
                    "bm42": models.SparseVector(
                        indices=sv["indices"], values=sv["values"]
                    ),
                },
                payload=payload,
            )
        )
    qdrant_utils.upsert_points(c, name, points)
    return len(points)


def index_scifact_deep_dive() -> dict[str, Any]:
    """Extra collections for scoring/quantization (SciFact only)."""
    corpus, queries, qrels = data.load_dataset("scifact")
    texts = [data.doc_text(r) for r in corpus]
    dense = embeddings.embed_dense(texts, query=False)
    sparse = embeddings.embed_sparse(texts)
    c = qdrant_utils.wait_ready()
    timings = {}
    resources = {}

    def points_dense():
        out = []
        for i, row in enumerate(corpus):
            out.append(
                models.PointStruct(
                    id=i,
                    vector=dense[i].tolist(),
                    payload={
                        "doc_id": str(row["_id"]),
                        "title": row.get("title") or "",
                        "text": row.get("text") or "",
                    },
                )
            )
        return out

    def points_sparse():
        out = []
        for i, row in enumerate(corpus):
            sv = sparse[i]
            out.append(
                models.PointStruct(
                    id=i,
                    vector={
                        "bm42": models.SparseVector(
                            indices=sv["indices"], values=sv["values"]
                        )
                    },
                    payload={"doc_id": str(row["_id"])},
                )
            )
        return out

    def timed(name, recreate, pts):
        t0 = time.perf_counter()
        recreate()
        qdrant_utils.upsert_points(c, name, pts)
        timings[name] = round(time.perf_counter() - t0, 3)
        resources[name] = qdrant_utils.collection_resources(c, name)

    pts_d = points_dense()
    pts_s = points_sparse()
    timed(
        config.COL_DENSE,
        lambda: qdrant_utils.recreate_dense(
            c, config.COL_DENSE, models.Distance.COSINE, on_disk=False
        ),
        pts_d,
    )
    timed(
        config.COL_BM42,
        lambda: qdrant_utils.recreate_sparse(c, config.COL_BM42, True, on_disk=False),
        pts_s,
    )
    timed(
        config.COL_DENSE_DOT,
        lambda: qdrant_utils.recreate_dense(c, config.COL_DENSE_DOT, models.Distance.DOT),
        pts_d,
    )
    timed(
        config.COL_DENSE_EUCLID,
        lambda: qdrant_utils.recreate_dense(
            c, config.COL_DENSE_EUCLID, models.Distance.EUCLID
        ),
        pts_d,
    )
    timed(
        config.COL_BM42_NO_IDF,
        lambda: qdrant_utils.recreate_sparse(c, config.COL_BM42_NO_IDF, False),
        pts_s,
    )
    scalar_q = models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, quantile=0.99, always_ram=True
        )
    )
    binary_q = models.BinaryQuantization(
        binary=models.BinaryQuantizationConfig(always_ram=True)
    )
    pq_q = models.ProductQuantization(
        product=models.ProductQuantizationConfig(
            compression=models.CompressionRatio.X16, always_ram=True
        )
    )
    timed(
        config.COL_Q_SCALAR,
        lambda: qdrant_utils.recreate_dense(
            c, config.COL_Q_SCALAR, models.Distance.COSINE, quantization=scalar_q
        ),
        pts_d,
    )
    timed(
        config.COL_Q_BINARY,
        lambda: qdrant_utils.recreate_dense(
            c, config.COL_Q_BINARY, models.Distance.COSINE, quantization=binary_q
        ),
        pts_d,
    )
    timed(
        config.COL_Q_PQ,
        lambda: qdrant_utils.recreate_dense(
            c, config.COL_Q_PQ, models.Distance.COSINE, quantization=pq_q
        ),
        pts_d,
    )
    return {
        "n_docs": len(corpus),
        "n_eval_queries": len(queries),
        "n_qrels": len(qrels),
        "index_seconds": timings,
        "resources": resources,
        "dense_model": config.DENSE_MODEL,
        "sparse_model": config.SPARSE_MODEL,
    }


# Back-compat name used by older run_all
def index_all() -> dict[str, Any]:
    return index_scifact_deep_dive()

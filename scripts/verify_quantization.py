"""Rebuild SciFact dense + quantized collections and compare metrics vs results/quantization.json."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from qdrant_client.http import models

from src import config, data, embeddings, experiments, metrics, qdrant_utils, search


def wait_indexed(c, name: str, timeout: float = 180) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        info = c.get_collection(name)
        pts = info.points_count or 0
        idx = info.indexed_vectors_count or 0
        status = str(info.status)
        if status == "green" and pts and idx >= max(0, pts - 200):
            return
        time.sleep(1)
    print(f"WARN: {name} not fully indexed: {c.get_collection(name)}")


def disk_breakdown(name: str) -> dict:
    storage = config.ROOT / "qdrant_storage" / "collections" / name
    rglob = 0
    if storage.exists():
        rglob = sum(p.stat().st_size for p in storage.rglob("*") if p.is_file())
    du_b = 0
    if storage.exists():
        out = subprocess.check_output(["du", "-sb", str(storage)], text=True)
        du_b = int(out.split()[0])
    return {
        "du_bytes": du_b,
        "du_mb": round(du_b / (1024 * 1024), 3),
        "rglob_bytes": rglob,
        "rglob_mb": round(rglob / (1024 * 1024), 3),
    }


def main() -> None:
    corpus, queries, qrels = data.load_dataset("scifact")
    texts = [data.doc_text(r) for r in corpus]
    print(f"Embedding {len(texts)} SciFact docs...")
    dense = embeddings.embed_dense(texts, query=False)
    c = qdrant_utils.wait_ready()

    pts = []
    for i, row in enumerate(corpus):
        pts.append(
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

    specs = [
        (config.COL_DENSE, None),
        (config.COL_Q_SCALAR, scalar_q),
        (config.COL_Q_BINARY, binary_q),
        (config.COL_Q_PQ, pq_q),
    ]
    for name, qcfg in specs:
        print(f"Recreating {name}...")
        qdrant_utils.recreate_dense(
            c, name, models.Distance.COSINE, quantization=qcfg, on_disk=False
        )
        batch = config.UPSERT_BATCH
        for i in range(0, len(pts), batch):
            qdrant_utils.upsert_points(c, name, pts[i : i + batch])
        wait_indexed(c, name)

    time.sleep(2)

    sizes = {}
    for name, _ in specs:
        info = c.get_collection(name)
        raw = json.loads(info.model_dump_json())
        sizes[name] = {
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": str(info.status),
            "segments_count": getattr(info, "segments_count", None),
            "quantization_config": raw.get("config", {}).get("quantization_config")
            if isinstance(raw.get("config"), dict)
            else raw.get("quantization_config"),
            **disk_breakdown(name),
        }
        print(
            f"{name}: du={sizes[name]['du_mb']} MB rglob={sizes[name]['rglob_mb']} MB "
            f"pts={sizes[name]['points_count']} indexed={sizes[name]['indexed_vectors_count']} "
            f"segs={sizes[name]['segments_count']}"
        )

    print("Embedding queries and evaluating...")
    q_dense, _ = experiments._query_cache(queries)
    variants = [
        (config.COL_DENSE, "full_fp32", None),
        (config.COL_Q_SCALAR, "scalar_int8_rescore", True),
        (config.COL_Q_SCALAR, "scalar_int8_no_rescore", False),
        (config.COL_Q_BINARY, "binary_rescore", True),
        (config.COL_Q_BINARY, "binary_no_rescore", False),
        (config.COL_Q_PQ, "pq_x16_rescore", True),
        (config.COL_Q_PQ, "pq_x16_no_rescore", False),
    ]
    runs = []
    for col, label, rescore in variants:
        params = None
        if rescore is not None:
            params = models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False, rescore=rescore, oversampling=2.0 if rescore else 1.0
                )
            )

        def retrieve(qid: str, text: str, col=col, params=params):
            i = next(j for j, q in enumerate(queries) if str(q["_id"]) == qid)
            ids, _, dt = search.search_dense(c, col, q_dense[i].tolist(), search_params=params)
            return ids, dt

        extra = {"collection": col, "rescore": rescore, **sizes[col]}
        print(f"Eval {label}...")
        runs.append(metrics.evaluate_run(label, retrieve, queries, qrels, extra=extra))

    old = json.loads((config.RESULTS_DIR / "quantization.json").read_text())
    old_by = {r["name"]: r for r in old["runs"]}
    comparison = []
    for r in runs:
        o = old_by[r["name"]]
        comparison.append(
            {
                "name": r["name"],
                "old_ndcg@10": o["metrics"]["ndcg@10"],
                "new_ndcg@10": r["metrics"]["ndcg@10"],
                "ndcg_delta": round(r["metrics"]["ndcg@10"] - o["metrics"]["ndcg@10"], 4),
                "old_disk_mb": o["extra"]["disk_mb"],
                "new_du_mb": r["extra"]["du_mb"],
                "new_rglob_mb": r["extra"]["rglob_mb"],
            }
        )

    out = {
        "sizes": sizes,
        "runs": runs,
        "comparison": comparison,
        "fp32_vector_mb": round(len(corpus) * config.DENSE_DIM * 4 / (1024 * 1024), 3),
    }
    path = config.RESULTS_DIR / "quantization_verify.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(comparison, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()

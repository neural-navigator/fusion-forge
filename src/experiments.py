from __future__ import annotations

import json
import time
from typing import Any

from qdrant_client.http import models

from . import config, data, embeddings, metrics, qdrant_utils, search


def _query_cache(queries: list[dict]):
    texts = [q["text"] for q in queries]
    return embeddings.cached_query_embeddings(texts)


def run_baseline() -> dict[str, Any]:
    corpus, queries, qrels = data.load_dataset()
    c = qdrant_utils.wait_ready()
    print(f"Embedding {len(queries)} evaluation queries...")
    q_dense, q_sparse = _query_cache(queries)

    runs = []

    def make_dense_retriever():
        def retrieve(qid: str, text: str):
            i = next(j for j, q in enumerate(queries) if str(q["_id"]) == qid)
            ids, _, dt = search.search_dense(
                c, config.COL_DENSE, q_dense[i].tolist()
            )
            return ids, dt

        return retrieve

    def make_bm42_retriever():
        def retrieve(qid: str, text: str):
            i = next(j for j, q in enumerate(queries) if str(q["_id"]) == qid)
            ids, _, dt = search.search_sparse(c, config.COL_BM42, q_sparse[i])
            return ids, dt

        return retrieve

    def make_hybrid_retriever(fusion: models.Fusion, name_col=config.COL_HYBRID):
        def retrieve(qid: str, text: str):
            i = next(j for j, q in enumerate(queries) if str(q["_id"]) == qid)
            ids, _, dt = search.search_fusion(
                c, name_col, q_dense[i].tolist(), q_sparse[i], fusion
            )
            return ids, dt

        return retrieve

    print("Evaluating dense (BGE-base, cosine)...")
    runs.append(
        metrics.evaluate_run(
            "dense_bge_base_cosine",
            make_dense_retriever(),
            queries,
            qrels,
            extra={"model": config.DENSE_MODEL, "metric": "cosine"},
        )
    )
    print("Evaluating BM42 (IDF)...")
    runs.append(
        metrics.evaluate_run(
            "bm42_idf",
            make_bm42_retriever(),
            queries,
            qrels,
            extra={"model": config.SPARSE_MODEL, "modifier": "IDF"},
        )
    )
    print("Evaluating hybrid RRF...")
    runs.append(
        metrics.evaluate_run(
            "hybrid_rrf",
            make_hybrid_retriever(models.Fusion.RRF),
            queries,
            qrels,
            extra={"fusion": "RRF", "prefetch": config.PREFETCH_LIMIT},
        )
    )

    out = {
        "dataset": config.DATASET,
        "n_docs": len(corpus),
        "n_queries": len(queries),
        "runs": runs,
    }
    _save("baseline.json", out)
    return out


def run_scoring() -> dict[str, Any]:
    _, queries, qrels = data.load_dataset()
    c = qdrant_utils.wait_ready()
    q_dense, q_sparse = _query_cache(queries)
    runs = []

    def dense_on(col: str, label: str, metric: str):
        def retrieve(qid: str, text: str):
            i = next(j for j, q in enumerate(queries) if str(q["_id"]) == qid)
            ids, _, dt = search.search_dense(c, col, q_dense[i].tolist())
            return ids, dt

        print(f"Scoring {label}...")
        runs.append(
            metrics.evaluate_run(
                label, retrieve, queries, qrels, extra={"metric": metric}
            )
        )

    dense_on(config.COL_DENSE, "dense_cosine", "cosine")
    if c.collection_exists(config.COL_DENSE_DOT):
        dense_on(config.COL_DENSE_DOT, "dense_dot", "dot")
    if c.collection_exists(config.COL_DENSE_EUCLID):
        dense_on(config.COL_DENSE_EUCLID, "dense_euclid", "euclid")

    def sparse_on(col: str, label: str, modifier: str):
        def retrieve(qid: str, text: str):
            i = next(j for j, q in enumerate(queries) if str(q["_id"]) == qid)
            ids, _, dt = search.search_sparse(c, col, q_sparse[i])
            return ids, dt

        print(f"Scoring {label}...")
        runs.append(
            metrics.evaluate_run(
                label, retrieve, queries, qrels, extra={"modifier": modifier}
            )
        )

    sparse_on(config.COL_BM42, "bm42_idf", "IDF")
    if c.collection_exists(config.COL_BM42_NO_IDF):
        sparse_on(config.COL_BM42_NO_IDF, "bm42_no_idf", "none")

    def fusion_on(fusion: models.Fusion, label: str):
        def retrieve(qid: str, text: str):
            i = next(j for j, q in enumerate(queries) if str(q["_id"]) == qid)
            ids, _, dt = search.search_fusion(
                c, config.COL_HYBRID, q_dense[i].tolist(), q_sparse[i], fusion
            )
            return ids, dt

        print(f"Scoring {label}...")
        runs.append(
            metrics.evaluate_run(
                label, retrieve, queries, qrels, extra={"fusion": fusion.name}
            )
        )

    fusion_on(models.Fusion.RRF, "hybrid_rrf")
    fusion_on(models.Fusion.DBSF, "hybrid_dbsf")

    weights = [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]
    for dw, sw in weights:
        label = f"hybrid_weighted_{dw}_{sw}"

        def retrieve(qid: str, text: str, dw=dw, sw=sw):
            i = next(j for j, q in enumerate(queries) if str(q["_id"]) == qid)
            t0 = time.perf_counter()
            d_ids, d_scores, _ = search.search_dense(
                c, config.COL_HYBRID, q_dense[i].tolist(), using="dense"
            )
            s_ids, s_scores, _ = search.search_sparse(
                c, config.COL_HYBRID, q_sparse[i], using="bm42"
            )
            ids = search.weighted_merge(d_ids, d_scores, s_ids, s_scores, dw, sw)
            return ids, time.perf_counter() - t0

        print(f"Scoring {label}...")
        runs.append(
            metrics.evaluate_run(
                label,
                retrieve,
                queries,
                qrels,
                extra={"dense_w": dw, "sparse_w": sw},
            )
        )

    out = {"runs": runs}
    _save("scoring.json", out)
    return out


def run_quantization() -> dict[str, Any]:
    _, queries, qrels = data.load_dataset()
    c = qdrant_utils.wait_ready()
    q_dense, _ = _query_cache(queries)
    runs = []
    resources = {}

    variants = [
        (config.COL_DENSE, "full_fp32", None),
        (config.COL_Q_SCALAR, "scalar_int8_rescore", True),
        (config.COL_Q_SCALAR, "scalar_int8_no_rescore", False),
        (config.COL_Q_BINARY, "binary_rescore", True),
        (config.COL_Q_BINARY, "binary_no_rescore", False),
        (config.COL_Q_PQ, "pq_x16_rescore", True),
        (config.COL_Q_PQ, "pq_x16_no_rescore", False),
    ]
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
            ids, _, dt = search.search_dense(
                c, col, q_dense[i].tolist(), search_params=params
            )
            return ids, dt

        print(f"Quantization eval {label}...")
        if not c.collection_exists(col):
            print(f"  skip missing {col}")
            continue
        extra = {"collection": col, "rescore": rescore}
        extra.update(qdrant_utils.collection_resources(c, col))
        resources[label] = extra
        runs.append(metrics.evaluate_run(label, retrieve, queries, qrels, extra=extra))

    out = {"runs": runs, "resources": resources}
    _save("quantization.json", out)
    return out


def _save(name: str, obj: Any) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / name
    path.write_text(json.dumps(obj, indent=2))
    print(f"Wrote {path}")

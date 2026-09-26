from __future__ import annotations

import json
import time
from typing import Any

from qdrant_client.http import models

from . import config, data, embeddings, metrics, qdrant_utils, search


def run_ranking() -> dict[str, Any]:
    corpus, queries, qrels = data.load_dataset()
    c = qdrant_utils.wait_ready()
    texts = [q["text"] for q in queries]
    q_dense, q_sparse = embeddings.cached_query_embeddings(texts)
    id_to_text = {str(r["_id"]): data.doc_text(r) for r in corpus}

    print("Loading cross-encoder", config.RERANKER_MODEL)
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(config.RERANKER_MODEL, device="cuda")

    runs = []

    def hybrid_rrf(qid: str, text: str):
        i = next(j for j, q in enumerate(queries) if str(q["_id"]) == qid)
        ids, _, dt = search.search_fusion(
            c, config.COL_HYBRID, q_dense[i].tolist(), q_sparse[i], models.Fusion.RRF
        )
        return ids, dt

    def hybrid_dbsf(qid: str, text: str):
        i = next(j for j, q in enumerate(queries) if str(q["_id"]) == qid)
        ids, _, dt = search.search_fusion(
            c, config.COL_HYBRID, q_dense[i].tolist(), q_sparse[i], models.Fusion.DBSF
        )
        return ids, dt

    def client_rrf(qid: str, text: str):
        i = next(j for j, q in enumerate(queries) if str(q["_id"]) == qid)
        t0 = time.perf_counter()
        d_ids, _, _ = search.search_dense(
            c, config.COL_HYBRID, q_dense[i].tolist(), using="dense"
        )
        s_ids, _, _ = search.search_sparse(
            c, config.COL_HYBRID, q_sparse[i], using="bm42"
        )
        ids = search.rrf_merge([d_ids, s_ids], k=60)
        return ids, time.perf_counter() - t0

    def ce_rerank(qid: str, text: str):
        i = next(j for j, q in enumerate(queries) if str(q["_id"]) == qid)
        t0 = time.perf_counter()
        ids, _, _ = search.search_fusion(
            c,
            config.COL_HYBRID,
            q_dense[i].tolist(),
            q_sparse[i],
            models.Fusion.RRF,
            limit=config.QUERY_K,
            prefetch=config.QUERY_K,
        )
        head, tail = ids[: config.RERANK_CANDIDATES], ids[config.RERANK_CANDIDATES :]
        pairs = [(text, id_to_text.get(did, "")) for did in head]
        scores = reranker.predict(pairs, batch_size=32)
        ranked = [did for did, _ in sorted(zip(head, scores), key=lambda x: x[1], reverse=True)]
        return ranked + tail, time.perf_counter() - t0

    def title_boost(qid: str, text: str):
        """Hybrid RRF with a light lexical title overlap boost (no extra model)."""
        i = next(j for j, q in enumerate(queries) if str(q["_id"]) == qid)
        t0 = time.perf_counter()
        ids, scores, _ = search.search_fusion(
            c, config.COL_HYBRID, q_dense[i].tolist(), q_sparse[i], models.Fusion.RRF
        )
        q_terms = set(text.lower().split())
        boosted = []
        for did, sc in zip(ids, scores):
            title = id_to_text.get(did, "").split(". ", 1)[0].lower()
            overlap = len(q_terms & set(title.split()))
            boosted.append((did, sc + 0.05 * overlap))
        boosted.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in boosted], time.perf_counter() - t0

    for name, fn, extra in [
        ("rank_hybrid_rrf", hybrid_rrf, {"method": "Qdrant Fusion.RRF"}),
        ("rank_hybrid_dbsf", hybrid_dbsf, {"method": "Qdrant Fusion.DBSF"}),
        ("rank_client_rrf_k60", client_rrf, {"method": "client RRF k=60 of dense+BM42"}),
        (
            "rank_ce_rerank_top50",
            ce_rerank,
            {"method": f"hybrid RRF top-{config.QUERY_K}, CE rerank first {config.RERANK_CANDIDATES}"},
        ),
        ("rank_title_overlap_boost", title_boost, {"method": "RRF + title term overlap"}),
    ]:
        print(f"Ranking eval {name}...")
        runs.append(metrics.evaluate_run(name, fn, queries, qrels, extra=extra))

    out = {
        "runs": runs,
        "recommendation": (
            "On SciFact, a MS MARCO MiniLM cross-encoder did not beat first-stage ranking "
            "(domain mismatch). Prefer dense BGE or dense-heavy weighted fusion (0.7/0.3). "
            "Use Qdrant RRF/DBSF when you need sparse recall; skip generic CE rerank unless "
            "the reranker is trained on the same domain."
        ),
    }
    (config.RESULTS_DIR / "ranking.json").write_text(json.dumps(out, indent=2))
    print("Wrote results/ranking.json")
    return out

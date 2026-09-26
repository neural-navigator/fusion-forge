from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from . import config, data, embeddings, metrics, qdrant_utils, search
from qdrant_client.http import models

ROUTE_PROTOTYPES = {
    "dense": (
        "A natural language question about meaning, claims, causality, or paraphrased "
        "scientific ideas. What happens, why, how does this theory explain."
    ),
    "bm42": (
        "Keyword lookup with rare technical terms, gene names, chemical names, "
        "abbreviations, exact phrases, and identifiers."
    ),
    "hybrid": (
        "A mixed query that names specific terms but also asks a conceptual question "
        "that needs both exact match and semantic similarity."
    ),
}

QUESTION_RE = re.compile(
    r"\b(what|why|how|does|do|is|are|can|could|should|whether)\b", re.I
)


def heuristic_scores(text: str) -> dict[str, float]:
    tokens = re.findall(r"[A-Za-z0-9_]+", text)
    n = max(len(tokens), 1)
    avg_len = sum(len(t) for t in tokens) / n
    has_q = 1.0 if (QUESTION_RE.search(text) or text.strip().endswith("?")) else 0.0
    caps = sum(1 for t in tokens if t.isupper() and len(t) >= 2) / n
    digits = sum(1 for t in tokens if any(ch.isdigit() for ch in t)) / n
    long_tokens = sum(1 for t in tokens if len(t) >= 10) / n
    bm42 = min(1.0, 0.15 + 1.6 * caps + 1.2 * digits + 0.8 * long_tokens + (0.2 if avg_len > 8 else 0))
    dense = min(1.0, 0.2 + 0.7 * has_q + (0.25 if n >= 8 else 0) - 0.4 * caps)
    hybrid = 0.45 + 0.2 * has_q
    return {"dense": max(dense, 0), "bm42": max(bm42, 0), "hybrid": hybrid}


def route_scores(text: str, proto_vecs: np.ndarray, labels: list[str], qvec: np.ndarray) -> dict[str, float]:
    q = qvec / (np.linalg.norm(qvec) + 1e-9)
    p = proto_vecs / (np.linalg.norm(proto_vecs, axis=1, keepdims=True) + 1e-9)
    sim = p @ q
    sim = (sim - sim.min()) / ((sim.max() - sim.min()) + 1e-9)
    embed_s = {lab: float(s) for lab, s in zip(labels, sim)}
    h = heuristic_scores(text)
    combined = {k: 0.55 * embed_s[k] + 0.45 * h[k] for k in labels}
    return {"embed": embed_s, "heuristic": h, "combined": combined}


def pick(scores: dict[str, float], margin: float = 0.06) -> str:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best, bscore = ranked[0]
    second = ranked[1][1]
    if best != "hybrid" and (bscore - second) < margin:
        return "hybrid"
    return best


def run_router() -> dict[str, Any]:
    _, queries, qrels = data.load_dataset()
    c = qdrant_utils.wait_ready()
    texts = [q["text"] for q in queries]
    q_dense, q_sparse = embeddings.cached_query_embeddings(texts)
    labels = list(ROUTE_PROTOTYPES)
    proto_vecs = embeddings.embed_dense([ROUTE_PROTOTYPES[k] for k in labels], query=True)

    backends = ("dense", "bm42", "hybrid")
    precomputed: dict[str, dict[str, list[str]]] = {b: {} for b in backends}
    latencies: dict[str, dict[str, float]] = {b: {} for b in backends}
    print("Precomputing backend rankings for router/oracle...")
    for i, q in enumerate(queries):
        qid = str(q["_id"])
        d_ids, _, dt_d = search.search_dense(c, config.COL_DENSE, q_dense[i].tolist())
        s_ids, _, dt_s = search.search_sparse(c, config.COL_BM42, q_sparse[i])
        h_ids, _, dt_h = search.search_fusion(
            c,
            config.COL_HYBRID,
            q_dense[i].tolist(),
            q_sparse[i],
            models.Fusion.RRF,
        )
        precomputed["dense"][qid] = d_ids
        precomputed["bm42"][qid] = s_ids
        precomputed["hybrid"][qid] = h_ids
        latencies["dense"][qid] = dt_d
        latencies["bm42"][qid] = dt_s
        latencies["hybrid"][qid] = dt_h

    def ndcg10(qid: str, ids: list[str]) -> float:
        return metrics.ndcg_at_k(ids, qrels[qid], 10)

    decisions = []
    for i, q in enumerate(queries):
        qid = str(q["_id"])
        text = q["text"]
        scores = route_scores(text, proto_vecs, labels, q_dense[i])
        choice_h = pick(scores["heuristic"])
        choice_e = pick(scores["embed"])
        choice_c = pick(scores["combined"])
        oracle = max(backends, key=lambda b: ndcg10(qid, precomputed[b][qid]))
        decisions.append(
            {
                "qid": qid,
                "text": text,
                "heuristic": choice_h,
                "embed": choice_e,
                "combined": choice_c,
                "oracle": oracle,
                "scores": scores,
                "ndcg": {b: round(ndcg10(qid, precomputed[b][qid]), 4) for b in backends},
            }
        )

    by_qid = {d["qid"]: d for d in decisions}

    def eval_policy(key: str, always: str | None = None) -> dict[str, Any]:
        def retrieve(qid: str, text: str):
            backend = always if always else by_qid[qid][key]
            return precomputed[backend][qid], latencies[backend][qid]

        run = metrics.evaluate_run(f"router_{always or key}", retrieve, queries, qrels)
        if always:
            counts = {always: len(queries)}
        else:
            counts = {b: sum(1 for d in decisions if d[key] == b) for b in backends}
        run["extra"] = {"route_counts": counts}
        return run

    runs = [
        eval_policy("", always="dense"),
        eval_policy("", always="bm42"),
        eval_policy("", always="hybrid"),
        eval_policy("heuristic"),
        eval_policy("embed"),
        eval_policy("combined"),
        eval_policy("oracle"),
    ]

    def confusion(key: str) -> dict[str, dict[str, int]]:
        table = {a: {b: 0 for b in backends} for a in backends}
        for d in decisions:
            table[d["oracle"]][d[key]] += 1
        return table

    out = {
        "runs": runs,
        "route_vs_oracle": {
            "heuristic": confusion("heuristic"),
            "embed": confusion("embed"),
            "combined": confusion("combined"),
        },
        "sample_decisions": [
            {k: d[k] for k in ("qid", "text", "heuristic", "embed", "combined", "oracle", "ndcg")}
            for d in decisions[:12]
        ],
        "notes": (
            "Routes send each query to dense (BGE), BM42, or hybrid RRF. "
            "Oracle picks the backend with the best nDCG@10 per query (upper bound)."
        ),
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "router.json").write_text(json.dumps(out, indent=2, default=str))
    print("Wrote results/router.json")
    return out

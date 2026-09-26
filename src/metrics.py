from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any


def dcg(rels: Sequence[float], k: int) -> float:
    s = 0.0
    for i, r in enumerate(rels[:k]):
        s += (2**r - 1) / math.log2(i + 2)
    return s


def ndcg_at_k(retrieved: Sequence[str], qrels: dict[str, int], k: int) -> float:
    rels = [qrels.get(did, 0) for did in retrieved[:k]]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    denom = dcg(ideal, k)
    return dcg(rels, k) / denom if denom else 0.0


def recall_at_k(retrieved: Sequence[str], qrels: dict[str, int], k: int) -> float:
    relevant = {d for d, r in qrels.items() if r > 0}
    if not relevant:
        return 0.0
    hit = sum(1 for d in retrieved[:k] if d in relevant)
    return hit / len(relevant)


def precision_at_k(retrieved: Sequence[str], qrels: dict[str, int], k: int) -> float:
    if k == 0:
        return 0.0
    hit = sum(1 for d in retrieved[:k] if qrels.get(d, 0) > 0)
    return hit / k


def mrr_at_k(retrieved: Sequence[str], qrels: dict[str, int], k: int) -> float:
    for i, did in enumerate(retrieved[:k], start=1):
        if qrels.get(did, 0) > 0:
            return 1.0 / i
    return 0.0


def ap_at_k(retrieved: Sequence[str], qrels: dict[str, int], k: int) -> float:
    relevant = {d for d, r in qrels.items() if r > 0}
    if not relevant:
        return 0.0
    hit = 0
    acc = 0.0
    for i, did in enumerate(retrieved[:k], start=1):
        if did in relevant:
            hit += 1
            acc += hit / i
    return acc / len(relevant)


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(sum(r[k] for r in rows) / len(rows)) for k in keys}


def evaluate_run(
    name: str,
    retrieve: Callable[[str, str], tuple[list[str], float]],
    queries: list[dict],
    qrels: dict[str, dict[str, int]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    per_query = []
    latencies = []
    for q in queries:
        qid = str(q["_id"])
        text = q["text"]
        if qid not in qrels:
            continue
        ids, latency = retrieve(qid, text)
        # BEIR default: drop the query document when it is also in the corpus (ArguAna, CQA, …).
        ids = [did for did in ids if did != qid]
        latencies.append(latency)
        rel = qrels[qid]
        per_query.append(
            {
                "ndcg@10": ndcg_at_k(ids, rel, 10),
                "ndcg@100": ndcg_at_k(ids, rel, 100),
                "recall@10": recall_at_k(ids, rel, 10),
                "recall@100": recall_at_k(ids, rel, 100),
                "precision@10": precision_at_k(ids, rel, 10),
                "mrr@10": mrr_at_k(ids, rel, 10),
                "map@100": ap_at_k(ids, rel, 100),
            }
        )
    metrics = aggregate(per_query)
    latencies_ms = [x * 1000 for x in latencies]
    latencies_ms.sort()

    def pct(p: float) -> float:
        if not latencies_ms:
            return 0.0
        idx = min(len(latencies_ms) - 1, int(round((p / 100) * (len(latencies_ms) - 1))))
        return latencies_ms[idx]

    return {
        "name": name,
        "n_queries": len(per_query),
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "latency_ms": {
            "mean": round(statistics.fmean(latencies_ms), 2) if latencies_ms else 0,
            "p50": round(pct(50), 2),
            "p95": round(pct(95), 2),
        },
        "extra": extra or {},
    }


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0

"""Router-gated sparse search + list-conditioned fusion + head rerank.

Pipeline (per query):

  1. Semantic router (BGE prototypes + lexical heuristic) decides whether BM42
     is worth running. Dense is always fetched. Sparse is skipped only when the
     router is confidently dense — that is the on/off switch.

  2. After candidates are in hand, list geometry (overlap, unique sparse head,
     top-1/top-2 margins) selects a scoring function or an adaptive dense prior
     α. No extra model: the two retrieved lists are the evidence.

  3. Head rerank: re-normalize scores on the fused top-H and apply the same
     operator again (sharper head). Optional title-term boost; optional
     cross-encoder on the head (usually domain-mismatched on SciFact).

Tuned on SciFact train (809), frozen on test (300).
"""
from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np

from . import config, data, embeddings, fusion_lab as fl, metrics, qdrant_utils, router, search

WINNER = {"kind": "rank_score", "params": {"a": 0.8, "mix": 0.75, "kappa": 20}}
HEAD = 50


def list_geometry(d_ids: list[str], d_sc: list[float], s_ids: list[str], s_sc: list[float]) -> dict[str, float]:
    nd = fl._minmax({i: s for i, s in zip(d_ids, d_sc)})
    ns = fl._minmax({i: s for i, s in zip(s_ids, s_sc)})
    d_m = (nd.get(d_ids[0], 1.0) - nd.get(d_ids[1], 0.0)) if len(d_ids) > 1 else 1.0
    s_m = (ns.get(s_ids[0], 1.0) - ns.get(s_ids[1], 0.0)) if len(s_ids) > 1 else 1.0
    ov10 = len(set(d_ids[:10]) & set(s_ids[:10])) / 10.0 if d_ids and s_ids else 0.0
    ov20 = len(set(d_ids[:20]) & set(s_ids[:20])) / 20.0 if d_ids and s_ids else 0.0
    d20 = set(d_ids[:20])
    s10 = s_ids[:10]
    sparse_unique = (sum(1 for d in s10 if d not in d20) / max(len(s10), 1)) if s10 else 0.0
    return {
        "ov10": ov10,
        "ov20": ov20,
        "d_margin": d_m,
        "s_margin": s_m,
        "sparse_unique": sparse_unique,
    }


def sparse_on(text: str, qvec: np.ndarray, proto_vecs: np.ndarray, labels: list[str], tau: float) -> tuple[bool, str, dict]:
    """Skip BM42 when tau < 0 (force dense), or when route is dense with gap ≥ tau.

    tau ≥ 9 means never skip. Dense search is always on; this is only the sparse switch.
    """
    scores = router.route_scores(text, proto_vecs, labels, qvec)
    choice = router.pick(scores["combined"])
    ranked = sorted(scores["combined"].items(), key=lambda x: x[1], reverse=True)
    gap = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 1.0
    if tau < 0:
        use = False
    else:
        use = not (choice == "dense" and gap >= tau)
    return use, choice, {"gap": round(gap, 4), "combined": scores["combined"]}


def adaptive_alpha(geo: dict[str, float], lex: float, coef: dict[str, float]) -> float:
    """Higher α → more dense. Overlap and dense margin raise α; unique sparse head and lexical BM42 lower it."""
    z = (
        coef["a0"]
        + coef["a_ov"] * geo["ov10"]
        + coef["a_m"] * math.tanh(geo["d_margin"] - geo["s_margin"])
        - coef["a_u"] * geo["sparse_unique"]
        - coef["a_lex"] * lex
    )
    return min(0.95, max(0.60, z))


def bucket_spec(geo: dict[str, float], t_ov: float, t_u: float, t_m: float) -> tuple[str, dict]:
    """Pick a frozen operator from the two lists (not from the query text)."""
    if geo["sparse_unique"] >= t_u:
        return "rank_score", {"a": 0.70, "mix": 0.75, "kappa": 20}
    if geo["ov10"] >= t_ov:
        return "minmax_linear", {"a": 0.88}
    if geo["d_margin"] >= t_m and geo["sparse_unique"] < 0.15:
        return "dense", {}
    return WINNER["kind"], dict(WINNER["params"])


def apply_fuse(row: dict, kind: str, params: dict) -> list[str]:
    return fl.fuse(row["d_ids"], row["d_sc"], row["s_ids"], row["s_sc"], kind, **params)


def head_rerank(row: dict, fused: list[str], kind: str, params: dict, head: int = HEAD) -> list[str]:
    keep = set(fused[:head])
    d_ids = [d for d in row["d_ids"] if d in keep]
    d_sc = [s for d, s in zip(row["d_ids"], row["d_sc"]) if d in keep]
    s_ids = [d for d in row["s_ids"] if d in keep]
    s_sc = [s for d, s in zip(row["s_ids"], row["s_sc"]) if d in keep]
    if kind == "dense" or not s_ids:
        new = fl.fuse(d_ids, d_sc, s_ids, s_sc, "dense")
    else:
        new = fl.fuse(d_ids, d_sc, s_ids, s_sc, kind, **params)
    seen = set(new)
    return new + [d for d in fused[head:] if d not in seen]


def title_rerank(text: str, ids: list[str], id_to_text: dict[str, str], head: int = HEAD) -> list[str]:
    q_terms = set(text.lower().split())
    top, tail = ids[:head], ids[head:]
    scored = []
    for rank, did in enumerate(top, 1):
        title = id_to_text.get(did, "").split(". ", 1)[0].lower()
        ov = len(q_terms & set(title.split()))
        scored.append((did, 1.0 / (20 + rank) + 0.06 * ov))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored] + tail


def retrieve_pack(queries, q_dense, q_sparse, c, col) -> list[dict]:
    out = []
    for i, q in enumerate(queries):
        qid = str(q["_id"])
        d_ids, d_sc, dt_d = search.search_dense(c, col, q_dense[i].tolist(), using="dense")
        s_ids, s_sc, dt_s = search.search_sparse(c, col, q_sparse[i], using="bm42")
        d_pair = [(d, s) for d, s in zip(d_ids, d_sc) if d != qid][: config.QUERY_K]
        s_pair = [(d, s) for d, s in zip(s_ids, s_sc) if d != qid][: config.QUERY_K]
        out.append(
            {
                "qid": qid,
                "text": q["text"],
                "qvec": q_dense[i],
                "d_ids": [p[0] for p in d_pair],
                "d_sc": [p[1] for p in d_pair],
                "s_ids": [p[0] for p in s_pair],
                "s_sc": [p[1] for p in s_pair],
                "dt_dense": dt_d,
                "dt_sparse": dt_s,
            }
        )
    return out


def rank_query(
    row: dict,
    proto_vecs,
    labels,
    tau: float,
    mode: str,
    coef: dict | None,
    bucket_t: tuple[float, float, float] | None,
    do_head: bool,
    do_title: bool,
    id_to_text: dict[str, str] | None,
) -> tuple[list[str], dict, float]:
    lex = router.heuristic_scores(row["text"])["bm42"]
    use_s, route, meta = sparse_on(row["text"], row["qvec"], proto_vecs, labels, tau)
    geo = list_geometry(row["d_ids"], row["d_sc"], row["s_ids"], row["s_sc"]) if use_s else {
        "ov10": 0.0, "ov20": 0.0, "d_margin": 1.0, "s_margin": 0.0, "sparse_unique": 0.0
    }
    if not use_s:
        ids = fl.fuse(row["d_ids"], row["d_sc"], row["s_ids"], row["s_sc"], "dense")
        kind, params = "dense", {}
    elif mode == "winner":
        kind, params = WINNER["kind"], dict(WINNER["params"])
        ids = apply_fuse(row, kind, params)
    elif mode == "alpha":
        a = adaptive_alpha(geo, lex, coef or {})
        kind, params = "rank_score", {"a": a, "mix": 0.75, "kappa": 20}
        ids = apply_fuse(row, kind, params)
    else:
        t_ov, t_u, t_m = bucket_t or (0.4, 0.3, 0.15)
        kind, params = bucket_spec(geo, t_ov, t_u, t_m)
        ids = apply_fuse(row, kind, params)
    if do_head and use_s:
        ids = head_rerank(row, ids, kind, params)
    if do_title and id_to_text:
        ids = title_rerank(row["text"], ids, id_to_text)
    info = {
        "sparse_on": use_s,
        "route": route,
        "kind": kind,
        "params": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()},
        **geo,
        **meta,
    }
    dt = row["dt_dense"] + (row["dt_sparse"] if use_s else 0.0)
    return ids, info, dt


def _eval_pack(pack, qrels, proto, labels, tau, mode, coef, bucket_t, do_head, do_title=False, id_to_text=None):
    vals, skip, kinds = [], 0, {}
    for row in pack:
        if row["qid"] not in qrels:
            continue
        ids, info, _ = rank_query(row, proto, labels, tau, mode, coef, bucket_t, do_head, do_title, id_to_text)
        vals.append(fl._ndcg10(ids, row["qid"], qrels))
        skip += 0 if info["sparse_on"] else 1
        kinds[info["kind"]] = kinds.get(info["kind"], 0) + 1
    return float(sum(vals) / len(vals)) if vals else 0.0, skip / max(len(vals), 1), kinds


def _test_run(name, pack, qrels, proto, labels, **kw):
    per, extras, skip = [], [], 0
    lats = []
    for row in pack:
        ids, info, dt = rank_query(row, proto, labels, **kw)
        ids = [d for d in ids if d != row["qid"]]
        rel = qrels[row["qid"]]
        per.append(
            {
                "ndcg@10": metrics.ndcg_at_k(ids, rel, 10),
                "ndcg@100": metrics.ndcg_at_k(ids, rel, 100),
                "recall@10": metrics.recall_at_k(ids, rel, 10),
                "recall@100": metrics.recall_at_k(ids, rel, 100),
                "precision@10": metrics.precision_at_k(ids, rel, 10),
                "mrr@10": metrics.mrr_at_k(ids, rel, 10),
                "map@100": metrics.ap_at_k(ids, rel, 100),
            }
        )
        extras.append({"qid": row["qid"], **info})
        skip += 0 if info["sparse_on"] else 1
        lats.append(dt * 1000)
    agg = metrics.aggregate(per)
    lats.sort()

    def pct(p):
        if not lats:
            return 0.0
        return lats[min(len(lats) - 1, int(round((p / 100) * (len(lats) - 1))))]

    kind_counts = {}
    for e in extras:
        kind_counts[e["kind"]] = kind_counts.get(e["kind"], 0) + 1
    return {
        "name": name,
        "n_queries": len(per),
        "metrics": {k: round(v, 4) for k, v in agg.items()},
        "latency_ms": {
            "mean": round(float(sum(lats) / len(lats)), 2) if lats else 0,
            "p50": round(pct(50), 2),
            "p95": round(pct(95), 2),
            "note": "Qdrant dense (+ sparse if gated on); excludes embed/rerank model time",
        },
        "extra": {
            "sparse_skip_rate": round(skip / max(len(per), 1), 4),
            "kind_counts": kind_counts,
            "sample": extras[:8],
        },
    }


def run_adaptive_pipeline(with_ce: bool = True) -> dict[str, Any]:
    train_q, train_qrels = data.load_scifact_split("train")
    test_q, test_qrels = data.load_scifact_split("test")
    corpus, _, _ = data.load_dataset()
    id_to_text = {str(r["_id"]): data.doc_text(r) for r in corpus}
    c = qdrant_utils.wait_ready()
    col = config.COL_HYBRID
    print(f"Embedding {len(train_q)} train + {len(test_q)} test queries...")
    tr_d, tr_s = embeddings.cached_query_embeddings([q["text"] for q in train_q])
    te_d, te_s = embeddings.cached_query_embeddings([q["text"] for q in test_q])
    labels = list(router.ROUTE_PROTOTYPES)
    proto = embeddings.embed_dense([router.ROUTE_PROTOTYPES[k] for k in labels], query=True)
    print("Retrieving dense and BM42 lists...")
    train_pack = retrieve_pack(train_q, tr_d, tr_s, c, col)
    test_pack = retrieve_pack(test_q, te_d, te_s, c, col)

    print("Tuning sparse-off margin and list-conditioned scorers on train...")
    winner_base, _, _ = _eval_pack(
        train_pack, train_qrels, proto, labels, 9.0, "winner", None, None, False
    )

    alpha_grid = []
    for a0 in (0.72, 0.78, 0.82):
        for a_ov in (0.0, 0.12, 0.24):
            for a_m in (0.0, 0.15, 0.30):
                for a_u in (0.0, 0.15, 0.30):
                    for a_lex in (0.0, 0.12):
                        coef = {"a0": a0, "a_ov": a_ov, "a_m": a_m, "a_u": a_u, "a_lex": a_lex}
                        sc, _, _ = _eval_pack(
                            train_pack, train_qrels, proto, labels, 9.0, "alpha", coef, None, False
                        )
                        alpha_grid.append({"coef": coef, "train_ndcg@10": round(sc, 4)})
    alpha_grid.sort(key=lambda r: r["train_ndcg@10"], reverse=True)
    best_alpha = alpha_grid[0]

    bucket_grid = []
    for t_ov in (0.3, 0.4, 0.5, 0.6):
        for t_u in (0.2, 0.3, 0.4):
            for t_m in (0.08, 0.15, 0.22):
                bt = (t_ov, t_u, t_m)
                sc, _, kinds = _eval_pack(
                    train_pack, train_qrels, proto, labels, 9.0, "bucket", None, bt, False
                )
                bucket_grid.append(
                    {"t_ov": t_ov, "t_u": t_u, "t_m": t_m, "train_ndcg@10": round(sc, 4), "kinds": kinds}
                )
    bucket_grid.sort(key=lambda r: r["train_ndcg@10"], reverse=True)
    best_bucket = bucket_grid[0]

    use_mode = "alpha" if best_alpha["train_ndcg@10"] >= best_bucket["train_ndcg@10"] else "bucket"
    coef = best_alpha["coef"]
    bt = (best_bucket["t_ov"], best_bucket["t_u"], best_bucket["t_m"])

    gate_grid = []
    for tau in (0.04, 0.06, 0.08, 0.12, 0.18, 9.0):
        sc, skip, _ = _eval_pack(
            train_pack, train_qrels, proto, labels, tau, use_mode, coef, bt, False
        )
        gate_grid.append({"tau": tau, "train_ndcg@10": round(sc, 4), "skip_rate": round(skip, 4)})
    gate_grid.sort(key=lambda r: (r["train_ndcg@10"], r["skip_rate"]), reverse=True)
    # Prefer a gate that does not lose nDCG vs never-skip; among those, more skip is fine.
    never = next(g for g in gate_grid if g["tau"] == 9.0)
    viable = [g for g in gate_grid if g["train_ndcg@10"] + 1e-9 >= never["train_ndcg@10"]]
    best_gate = max(viable, key=lambda g: g["skip_rate"]) if viable else never

    head_on, _, _ = _eval_pack(
        train_pack, train_qrels, proto, labels, best_gate["tau"], use_mode, coef, bt, True
    )
    title_on, _, _ = _eval_pack(
        train_pack,
        train_qrels,
        proto,
        labels,
        best_gate["tau"],
        use_mode,
        coef,
        bt,
        True,
        True,
        id_to_text,
    )
    do_head = head_on >= never["train_ndcg@10"] - 0.0005
    do_title = title_on >= (head_on if do_head else never["train_ndcg@10"]) - 0.0005

    kw = dict(
        tau=best_gate["tau"],
        mode=use_mode,
        coef=coef,
        bucket_t=bt,
        do_head=do_head,
        do_title=do_title,
        id_to_text=id_to_text if do_title else None,
    )

    test_runs = [
        _test_run(
            "dense_only",
            test_pack,
            test_qrels,
            proto,
            labels,
            tau=-1.0,
            mode="winner",
            coef=None,
            bucket_t=None,
            do_head=False,
            do_title=False,
            id_to_text=None,
        ),
        _test_run(
            "always_sparse_winner_formula",
            test_pack,
            test_qrels,
            proto,
            labels,
            tau=9.0,
            mode="winner",
            coef=None,
            bucket_t=None,
            do_head=False,
            do_title=False,
            id_to_text=None,
        ),
        _test_run(
            "router_gate_winner_formula",
            test_pack,
            test_qrels,
            proto,
            labels,
            tau=best_gate["tau"],
            mode="winner",
            coef=None,
            bucket_t=None,
            do_head=False,
            do_title=False,
            id_to_text=None,
        ),
        _test_run(
            f"list_scorer_{use_mode}_always_sparse",
            test_pack,
            test_qrels,
            proto,
            labels,
            tau=9.0,
            mode=use_mode,
            coef=coef,
            bucket_t=bt,
            do_head=False,
            do_title=False,
            id_to_text=None,
        ),
        _test_run("router_plus_list_scorer", test_pack, test_qrels, proto, labels, **{**kw, "do_head": False, "do_title": False, "id_to_text": None}),
        _test_run("full_pipeline", test_pack, test_qrels, proto, labels, **kw),
    ]

    # dense_only used tau=0 which skips ALL sparse — correct. But mode winner with sparse off uses dense fuse. Good.

    if with_ce:
        print("Cross-encoder rerank of fused head on test...")
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(config.RERANKER_MODEL, device="cuda")
        per, lats = [], []
        for row in test_pack:
            t0 = time.perf_counter()
            ids, info, dt = rank_query(row, proto, labels, **kw)
            head, tail = ids[:HEAD], ids[HEAD:]
            pairs = [(row["text"], id_to_text.get(did, "")) for did in head]
            scores = reranker.predict(pairs, batch_size=32)
            ranked = [did for did, _ in sorted(zip(head, scores), key=lambda x: x[1], reverse=True)]
            ids = [d for d in ranked + tail if d != row["qid"]]
            rel = test_qrels[row["qid"]]
            per.append(
                {
                    "ndcg@10": metrics.ndcg_at_k(ids, rel, 10),
                    "ndcg@100": metrics.ndcg_at_k(ids, rel, 100),
                    "recall@10": metrics.recall_at_k(ids, rel, 10),
                    "recall@100": metrics.recall_at_k(ids, rel, 100),
                    "precision@10": metrics.precision_at_k(ids, rel, 10),
                    "mrr@10": metrics.mrr_at_k(ids, rel, 10),
                    "map@100": metrics.ap_at_k(ids, rel, 100),
                }
            )
            lats.append((time.perf_counter() - t0) * 1000)
        agg = metrics.aggregate(per)
        lats.sort()
        test_runs.append(
            {
                "name": "full_pipeline_ce_head",
                "n_queries": len(per),
                "metrics": {k: round(v, 4) for k, v in agg.items()},
                "latency_ms": {
                    "mean": round(float(sum(lats) / len(lats)), 2),
                    "p50": round(lats[len(lats) // 2], 2),
                    "p95": round(lats[min(len(lats) - 1, int(0.95 * (len(lats) - 1)))], 2),
                    "note": "includes CE predict time",
                },
                "extra": {"reranker": config.RERANKER_MODEL, "head": HEAD},
            }
        )

    test_runs.sort(key=lambda r: r["metrics"]["ndcg@10"], reverse=True)

    out = {
        "dataset": "scifact",
        "protocol": (
            "Train (809) selects sparse-off margin, list-conditioned scorer, and whether "
            "head/title rerank help. Frozen on test (300). Dense is always retrieved; BM42 "
            "is skipped when the semantic router is confidently dense."
        ),
        "stages": {
            "router_sparse_gate": (
                "combined prototype+heuristic route; skip sparse iff route=dense and "
                "score gap ≥ tau. Dense search always on (BM42-only is worse on SciFact)."
            ),
            "list_conditioned_scorer": (
                "From the two lists: overlap@10, unique sparse-in-top-10, top-1/top-2 "
                "margins. Either a piecewise operator (bucket) or α(list) in the rank–score mix "
                "u = mix(α s̃_d + (1-α) s̃_s) + (1-mix)(α/(κ+r_d)+(1-α)/(κ+r_s))."
            ),
            "head_rerank": (
                f"Re-minmax the fused top-{HEAD} and apply the same operator (sharper head). "
                "Optional title-term overlap. Optional MS MARCO MiniLM CE (ablation)."
            ),
        },
        "train": {
            "winner_formula_ndcg@10": round(winner_base, 4),
            "best_alpha": best_alpha,
            "best_bucket": best_bucket,
            "selected_mode": use_mode,
            "gate_grid": gate_grid,
            "best_gate": best_gate,
            "head_rerank_train_ndcg@10": round(head_on, 4),
            "title_rerank_train_ndcg@10": round(title_on, 4),
            "use_head_rerank": do_head,
            "use_title_rerank": do_title,
        },
        "test_runs": test_runs,
        "winner": test_runs[0]["name"],
    }
    path = config.RESULTS_DIR / "adaptive_pipeline.json"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str))
    print("Winner", out["winner"], test_runs[0]["metrics"]["ndcg@10"])
    print("Wrote", path)
    return out


if __name__ == "__main__":
    run_adaptive_pipeline()

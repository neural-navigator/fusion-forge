"""
Dense cosine and BM42 IDF are not commensurate: one is a bounded similarity,
the other an unbounded weighted sum. Fusion is therefore:

  1. Map each channel to a utility in [0, 1] (rank, min-max, z, softmax).
  2. Combine utilities with an operator that encodes a dependence assumption
     (convex mix, CombMNZ / two-hits, noisy-OR / independent evidence).
  3. Optionally let the mix weight depend on a query-level confidence gate
     (top-1 vs top-2 margin, list overlap, lexical cues).
  4. Optionally calibrate a linear model of those features on train qrels.

Operators are selected on SciFact train (809 queries) and frozen on test (300).
"""
from __future__ import annotations

import json
import math
import time
from typing import Any, Callable

import numpy as np

from . import config, data, embeddings, metrics, qdrant_utils, router, search

K = config.QUERY_K
MISS_RANK = K + 1


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    span = (hi - lo) or 1.0
    return {i: (s - lo) / span for i, s in scores.items()}


def _z01(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=np.float64)
    mu, sd = float(vals.mean()), float(vals.std() or 1.0)
    z = {i: (s - mu) / sd for i, s in scores.items()}
    lo, hi = min(z.values()), max(z.values())
    span = (hi - lo) or 1.0
    return {i: (v - lo) / span for i, v in z.items()}


def _softmax(scores: dict[str, float], temp: float) -> dict[str, float]:
    if not scores:
        return {}
    xs = np.array(list(scores.values()), dtype=np.float64) / max(temp, 1e-6)
    xs = xs - xs.max()
    e = np.exp(xs)
    e = e / (e.sum() or 1.0)
    return {i: float(p) for i, p in zip(scores, e)}


def _rrf(rank: int, kappa: float) -> float:
    return 1.0 / (kappa + rank)


def _maps(ids: list[str], scores: list[float]) -> tuple[dict[str, int], dict[str, float]]:
    ranks = {did: i + 1 for i, did in enumerate(ids)}
    sc = {did: float(s) for did, s in zip(ids, scores)}
    return ranks, sc


def _universe(d_ids: list[str], s_ids: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for did in d_ids + s_ids:
        seen.setdefault(did, None)
    return list(seen)


def _sort(util: dict[str, float], limit: int = K) -> list[str]:
    return [d for d, _ in sorted(util.items(), key=lambda x: x[1], reverse=True)[:limit]]


def fuse(
    d_ids: list[str],
    d_sc: list[float],
    s_ids: list[str],
    s_sc: list[float],
    kind: str,
    **p: Any,
) -> list[str]:
    rd, sd = _maps(d_ids, d_sc)
    rs, ss = _maps(s_ids, s_sc)
    keys = _universe(d_ids, s_ids)
    nd, ns = _minmax(sd), _minmax(ss)
    zd, zs = _z01(sd), _z01(ss)
    kappa = float(p.get("kappa", 60))
    a = float(p.get("a", 0.75))
    b = 1.0 - a
    gamma = float(p.get("gamma", 0.15))
    power = float(p.get("power", 1.0))
    temp = float(p.get("temp", 0.08))
    pd, ps = _softmax(sd, temp), _softmax(ss, temp)

    util: dict[str, float] = {}
    for did in keys:
        r1, r2 = rd.get(did, MISS_RANK), rs.get(did, MISS_RANK)
        x = nd.get(did, 0.0)
        y = ns.get(did, 0.0)
        both = 1.0 if (did in rd and did in rs) else 0.0
        hits = (1.0 if did in rd else 0.0) + (1.0 if did in rs else 0.0)
        if kind == "dense":
            util[did] = x if did in rd else -1.0
        elif kind == "sparse":
            util[did] = y if did in rs else -1.0
        elif kind == "rrf":
            util[did] = a * _rrf(r1, kappa) + b * _rrf(r2, kappa)
        elif kind == "isr":
            util[did] = a / (r1 * r1) + b / (r2 * r2)
        elif kind == "minmax_linear":
            util[did] = a * x + b * y
        elif kind == "z_linear":
            util[did] = a * zd.get(did, 0.0) + b * zs.get(did, 0.0)
        elif kind == "softmax_linear":
            util[did] = a * pd.get(did, 0.0) + b * ps.get(did, 0.0)
        elif kind == "combmnz":
            util[did] = hits * (x + y)
        elif kind == "combmax":
            util[did] = max(x, y)
        elif kind == "noisy_or":
            util[did] = 1.0 - (1.0 - pd.get(did, 0.0)) * (1.0 - ps.get(did, 0.0))
        elif kind == "geom":
            util[did] = ((x + 1e-6) ** a) * ((y + 1e-6) ** b)
        elif kind == "harmonic":
            util[did] = 0.0 if (x + y) == 0 else 2 * x * y / (x + y)
        elif kind == "power_mean":
            util[did] = (a * (x + 1e-9) ** power + b * (y + 1e-9) ** power) ** (1.0 / power)
        elif kind == "rank_score":
            mix = float(p.get("mix", 0.5))
            util[did] = mix * (a * x + b * y) + (1.0 - mix) * (
                a * _rrf(r1, kappa) + b * _rrf(r2, kappa)
            )
        elif kind == "overlap_boost":
            util[did] = a * x + b * y + gamma * both * min(x, y)
        elif kind == "product_evidence":
            util[did] = a * x + b * y + gamma * x * y
        else:
            raise ValueError(kind)
    return _sort(util)


def query_gate_alpha(
    d_ids: list[str],
    d_sc: list[float],
    s_ids: list[str],
    s_sc: list[float],
    text: str,
    w_margin: float,
    w_lex: float,
    w_ov: float,
    base: float = 0.78,
) -> float:
    """Higher alpha → more dense. Margin, lexical BM42-ness, top-10 overlap."""
    nd = _minmax({i: s for i, s in zip(d_ids, d_sc)})
    ns = _minmax({i: s for i, s in zip(s_ids, s_sc)})
    d_m = (nd.get(d_ids[0], 1.0) - nd.get(d_ids[1], 0.0)) if len(d_ids) > 1 else 1.0
    s_m = (ns.get(s_ids[0], 1.0) - ns.get(s_ids[1], 0.0)) if len(s_ids) > 1 else 1.0
    ov = len(set(d_ids[:10]) & set(s_ids[:10])) / 10.0
    lex = router.heuristic_scores(text)["bm42"]
    z = w_margin * (d_m - s_m) - w_lex * lex - w_ov * ov
    a = base + 0.18 * math.tanh(z)
    return min(0.95, max(0.55, a))


def fuse_gated(d_ids, d_sc, s_ids, s_sc, text, w_margin, w_lex, w_ov, extra_gamma=0.12):
    a = query_gate_alpha(d_ids, d_sc, s_ids, s_sc, text, w_margin, w_lex, w_ov)
    return fuse(
        d_ids, d_sc, s_ids, s_sc, "overlap_boost", a=a, gamma=extra_gamma
    ), a


def fuse_margin_switch(d_ids, d_sc, s_ids, s_sc, tau: float, a: float):
    nd = _minmax({i: s for i, s in zip(d_ids, d_sc)})
    d_m = (nd.get(d_ids[0], 1.0) - nd.get(d_ids[1], 0.0)) if len(d_ids) > 1 else 1.0
    if d_m >= tau:
        return fuse(d_ids, d_sc, s_ids, s_sc, "dense")
    return fuse(d_ids, d_sc, s_ids, s_sc, "overlap_boost", a=a, gamma=0.12)


def _ndcg10(ids: list[str], qid: str, qrels: dict[str, dict[str, int]]) -> float:
    ids = [d for d in ids if d != qid]
    return metrics.ndcg_at_k(ids, qrels[qid], 10)


def mean_ndcg(pack: list[dict], qrels, ranker: Callable[[dict], list[str]]) -> float:
    vals = []
    for row in pack:
        qid = row["qid"]
        if qid not in qrels:
            continue
        vals.append(_ndcg10(ranker(row), qid, qrels))
    return float(sum(vals) / len(vals)) if vals else 0.0


def retrieve_pack(queries: list[dict], q_dense, q_sparse, c, col: str) -> list[dict]:
    out = []
    for i, q in enumerate(queries):
        qid = str(q["_id"])
        d_ids, d_sc, _ = search.search_dense(c, col, q_dense[i].tolist(), using="dense")
        s_ids, s_sc, _ = search.search_sparse(c, col, q_sparse[i], using="bm42")
        d_pair = [(d, s) for d, s in zip(d_ids, d_sc) if d != qid][:K]
        s_pair = [(d, s) for d, s in zip(s_ids, s_sc) if d != qid][:K]
        d_ids, d_sc = [p[0] for p in d_pair], [p[1] for p in d_pair]
        s_ids, s_sc = [p[0] for p in s_pair], [p[1] for p in s_pair]
        out.append(
            {
                "qid": qid,
                "text": q["text"],
                "d_ids": d_ids[:K],
                "d_sc": d_sc[:K],
                "s_ids": s_ids[:K],
                "s_sc": s_sc[:K],
            }
        )
    return out


def _candidate_features(row: dict, did: str) -> np.ndarray:
    rd, sd = _maps(row["d_ids"], row["d_sc"])
    rs, ss = _maps(row["s_ids"], row["s_sc"])
    nd, ns = _minmax(sd), _minmax(ss)
    x, y = nd.get(did, 0.0), ns.get(did, 0.0)
    r1, r2 = rd.get(did, MISS_RANK), rs.get(did, MISS_RANK)
    both = 1.0 if (did in rd and did in rs) else 0.0
    return np.array(
        [
            1.0,
            x,
            y,
            _rrf(r1, 60),
            _rrf(r2, 60),
            both,
            x * y,
            math.log(r1),
            math.log(r2),
            1.0 / (r1 * r1),
            1.0 / (r2 * r2),
        ],
        dtype=np.float64,
    )


def fit_logistic(train_pack: list[dict], qrels, steps: int = 250, lr: float = 0.4, l2: float = 0.02) -> np.ndarray:
    xs, ys, ws = [], [], []
    for row in train_pack:
        rel = {d for d, r in qrels[row["qid"]].items() if r > 0}
        universe = _universe(row["d_ids"], row["s_ids"])
        for did in universe:
            xs.append(_candidate_features(row, did))
            y = 1.0 if did in rel else 0.0
            ys.append(y)
            ws.append(8.0 if y else 1.0)
    X = np.vstack(xs)
    y = np.array(ys)
    wgt = np.array(ws)
    wgt = wgt / wgt.mean()
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        grad = (X.T @ (wgt * (p - y))) / len(y) + l2 * w
        w -= lr * grad
    return w


def fuse_logistic(row: dict, w: np.ndarray) -> list[str]:
    util = {}
    for did in _universe(row["d_ids"], row["s_ids"]):
        z = float(_candidate_features(row, did) @ w)
        util[did] = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, z))))
    return _sort(util)


def formula_catalog() -> list[tuple[str, dict]]:
    specs: list[tuple[str, dict]] = [
        ("dense", {}),
        ("sparse", {}),
        ("rrf", {"a": 0.5, "kappa": 60}),
        ("rrf", {"a": 0.7, "kappa": 60}),
        ("rrf", {"a": 0.8, "kappa": 60}),
        ("rrf", {"a": 0.7, "kappa": 20}),
        ("rrf", {"a": 0.7, "kappa": 10}),
        ("isr", {"a": 0.75}),
        ("isr", {"a": 0.85}),
        ("combmax", {}),
        ("combmnz", {}),
        ("harmonic", {}),
        ("noisy_or", {"temp": 0.05}),
        ("noisy_or", {"temp": 0.08}),
        ("noisy_or", {"temp": 0.15}),
    ]
    for a in (0.55, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        specs.append(("minmax_linear", {"a": a}))
        specs.append(("z_linear", {"a": a}))
        specs.append(("softmax_linear", {"a": a, "temp": 0.08}))
        specs.append(("geom", {"a": a}))
        specs.append(("power_mean", {"a": a, "power": 2.0}))
        specs.append(("overlap_boost", {"a": a, "gamma": 0.12}))
        specs.append(("product_evidence", {"a": a, "gamma": 0.25}))
        specs.append(("rank_score", {"a": a, "mix": 0.55, "kappa": 60}))
        specs.append(("rank_score", {"a": a, "mix": 0.75, "kappa": 20}))
    return specs


def _spec_name(kind: str, p: dict) -> str:
    if not p:
        return kind
    bits = ",".join(f"{k}={v}" for k, v in p.items())
    return f"{kind}({bits})"


def run_fusion_lab() -> dict[str, Any]:
    train_q, train_qrels = data.load_scifact_split("train")
    test_q, test_qrels = data.load_scifact_split("test")
    c = qdrant_utils.wait_ready()
    col = config.COL_HYBRID
    print(f"Embedding {len(train_q)} train + {len(test_q)} test queries...")
    tr_d, tr_s = embeddings.cached_query_embeddings([q["text"] for q in train_q])
    te_d, te_s = embeddings.cached_query_embeddings([q["text"] for q in test_q])
    print("Retrieving dense and BM42 lists...")
    train_pack = retrieve_pack(train_q, tr_d, tr_s, c, col)
    test_pack = retrieve_pack(test_q, te_d, te_s, c, col)

    def eval_spec(pack, qrels, kind, p):
        return mean_ndcg(
            pack,
            qrels,
            lambda row: fuse(row["d_ids"], row["d_sc"], row["s_ids"], row["s_sc"], kind, **p),
        )

    print("Grid-search fusion operators on train...")
    train_scores = []
    for kind, p in formula_catalog():
        sc = eval_spec(train_pack, train_qrels, kind, p)
        train_scores.append({"name": _spec_name(kind, p), "kind": kind, "params": p, "train_ndcg@10": round(sc, 4)})
    train_scores.sort(key=lambda r: r["train_ndcg@10"], reverse=True)

    print("Tuning query gate on train...")
    gate_grid = []
    for wm in (0.0, 0.8, 1.6, 2.4):
        for wl in (0.0, 0.6, 1.2):
            for wo in (0.0, 0.8, 1.6):
                sc = mean_ndcg(
                    train_pack,
                    train_qrels,
                    lambda row, wm=wm, wl=wl, wo=wo: fuse_gated(
                        row["d_ids"], row["d_sc"], row["s_ids"], row["s_sc"], row["text"], wm, wl, wo
                    )[0],
                )
                gate_grid.append({"w_margin": wm, "w_lex": wl, "w_ov": wo, "train_ndcg@10": round(sc, 4)})
    gate_grid.sort(key=lambda r: r["train_ndcg@10"], reverse=True)
    best_gate = gate_grid[0]

    print("Tuning dense-margin switch on train...")
    switch_grid = []
    for tau in (0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25):
        for a in (0.75, 0.82, 0.88):
            sc = mean_ndcg(
                train_pack,
                train_qrels,
                lambda row, tau=tau, a=a: fuse_margin_switch(
                    row["d_ids"], row["d_sc"], row["s_ids"], row["s_sc"], tau, a
                ),
            )
            switch_grid.append({"tau": tau, "a": a, "train_ndcg@10": round(sc, 4)})
    switch_grid.sort(key=lambda r: r["train_ndcg@10"], reverse=True)
    best_switch = switch_grid[0]

    print("Fitting logistic evidence model on train...")
    t0 = time.perf_counter()
    w = fit_logistic(train_pack, train_qrels)
    logit_train = mean_ndcg(train_pack, train_qrels, lambda row: fuse_logistic(row, w))
    logit_sec = round(time.perf_counter() - t0, 2)

    def test_run(name, ranker, extra=None):
        per = []
        for row in test_pack:
            ids = [d for d in ranker(row) if d != row["qid"]]
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
        agg = metrics.aggregate(per)
        return {
            "name": name,
            "n_queries": len(per),
            "metrics": {k: round(v, 4) for k, v in agg.items()},
            "extra": extra or {},
        }

    frozen = train_scores[:8]
    test_runs = [
        test_run("dense", lambda r: fuse(r["d_ids"], r["d_sc"], r["s_ids"], r["s_sc"], "dense")),
        test_run("bm42", lambda r: fuse(r["d_ids"], r["d_sc"], r["s_ids"], r["s_sc"], "sparse")),
        test_run(
            "rrf_equal_k60",
            lambda r: fuse(r["d_ids"], r["d_sc"], r["s_ids"], r["s_sc"], "rrf", a=0.5, kappa=60),
        ),
    ]
    for row in frozen:
        test_runs.append(
            test_run(
                "trainbest_" + row["name"],
                lambda r, k=row["kind"], p=row["params"]: fuse(
                    r["d_ids"], r["d_sc"], r["s_ids"], r["s_sc"], k, **p
                ),
                extra={"selected_on": "scifact_train", **row},
            )
        )
    test_runs.append(
        test_run(
            "gated_overlap",
            lambda r: fuse_gated(
                r["d_ids"],
                r["d_sc"],
                r["s_ids"],
                r["s_sc"],
                r["text"],
                best_gate["w_margin"],
                best_gate["w_lex"],
                best_gate["w_ov"],
            )[0],
            extra={"selected_on": "scifact_train", **best_gate},
        )
    )
    test_runs.append(
        test_run(
            "margin_switch",
            lambda r: fuse_margin_switch(
                r["d_ids"],
                r["d_sc"],
                r["s_ids"],
                r["s_sc"],
                best_switch["tau"],
                best_switch["a"],
            ),
            extra={"selected_on": "scifact_train", **best_switch},
        )
    )
    test_runs.append(
        test_run(
            "logistic_evidence",
            lambda r: fuse_logistic(r, w),
            extra={
                "selected_on": "scifact_train",
                "train_ndcg@10": round(logit_train, 4),
                "weights": [round(float(x), 4) for x in w],
                "fit_seconds": logit_sec,
                "features": [
                    "bias",
                    "minmax_dense",
                    "minmax_sparse",
                    "rrf_dense",
                    "rrf_sparse",
                    "in_both",
                    "product",
                    "log_rank_d",
                    "log_rank_s",
                    "isr_d",
                    "isr_s",
                ],
            },
        )
    )

    # Per-query oracle over α in minmax_linear (upper bound for convex score mix).
    oracle = []
    alphas = [i / 20 for i in range(21)]
    for row in test_pack:
        best = -1.0
        for a in alphas:
            ids = fuse(row["d_ids"], row["d_sc"], row["s_ids"], row["s_sc"], "minmax_linear", a=a)
            best = max(best, _ndcg10(ids, row["qid"], test_qrels))
        oracle.append(best)
    oracle_ndcg = round(float(sum(oracle) / len(oracle)), 4)

    test_runs.sort(key=lambda r: r["metrics"]["ndcg@10"], reverse=True)
    dense_n = next(r for r in test_runs if r["name"] == "dense")["metrics"]["ndcg@10"]
    rrf_n = next(r for r in test_runs if r["name"] == "rrf_equal_k60")["metrics"]["ndcg@10"]
    winner = test_runs[0]

    out = {
        "dataset": "scifact",
        "protocol": (
            "Tune on SciFact train (809 queries), freeze, evaluate on test (300). "
            "Same hybrid collection lists (dense named vector + BM42 IDF), k=100, "
            "ignore_identical_ids."
        ),
        "n_train": len(train_pack),
        "n_test": len(test_pack),
        "theory": {
            "incommensurable": (
                "Cosine is bounded similarity; BM42 is an unbounded IDF-weighted sum. "
                "Raw addition is meaningless; each channel is mapped to a utility in [0,1] "
                "or to a rank kernel 1/(κ+r)."
            ),
            "operators": (
                "Convex mix = interchangeable utilities. CombMNZ / product / noisy-OR "
                "reward agreement (two independent likelihoods). ISR 1/r^2 concentrates "
                "on the head. A query gate moves the dense prior using top-1/top-2 margin, "
                "lexical BM42-ness, and top-10 overlap."
            ),
            "logistic": (
                "P(rel|d) ≈ σ(w·φ) with φ = (minmax, RRF, overlap, product, log-rank, ISR) "
                "fit by weighted logistic regression on train candidates in the union of lists."
            ),
        },
        "train_top": train_scores[:15],
        "best_gate_train": best_gate,
        "best_switch_train": best_switch,
        "logistic_train_ndcg@10": round(logit_train, 4),
        "test_runs": test_runs,
        "oracle_minmax_alpha_ndcg@10": oracle_ndcg,
        "beats_dense": winner["metrics"]["ndcg@10"] > dense_n,
        "beats_rrf": winner["metrics"]["ndcg@10"] > rrf_n,
        "winner": winner["name"],
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / "fusion_lab.json"
    path.write_text(json.dumps(out, indent=2))
    print("Winner", winner["name"], winner["metrics"]["ndcg@10"], "dense", dense_n, "rrf", rrf_n)
    print("Wrote", path)
    return out


if __name__ == "__main__":
    run_fusion_lab()

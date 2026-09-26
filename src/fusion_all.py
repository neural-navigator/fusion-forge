"""Train-then-freeze the parameterized hybrid mix on every indexed BEIR unit.

Protocol:
  - If official train.tsv exists: tune on train, evaluate on test.tsv.
  - Else if dev.tsv exists: tune on dev, evaluate on test.
  - Else: hold out 30% of test queries (seed 42; 50/50 if n<40).

Search the same family as the SciFact study: dense, BM42, equal RRF, min-max
linear α, rank–score (α, λ=0.75, κ=20). Best train nDCG@10 is frozen for test.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import config, data, embeddings, fusion_lab as fl, metrics, qdrant_utils, search

SEED = 42
PATH = config.RESULTS_DIR / "fusion_all.json"

SPECS: list[tuple[str, dict]] = [("dense", {}), ("sparse", {}), ("rrf", {"a": 0.5, "kappa": 60})]
for a in (0.50, 0.60, 0.70, 0.80, 0.90, 1.00):
    SPECS.append(("minmax_linear", {"a": a}))
for a in (0.60, 0.70, 0.80, 0.90):
    SPECS.append(("rank_score", {"a": a, "mix": 0.75, "kappa": 20}))


def _spec_name(kind: str, p: dict) -> str:
    if not p:
        return kind
    return kind + "(" + ",".join(f"{k}={v}" for k, v in p.items()) + ")"


def load_tune_test(dataset: str) -> dict:
    zname = data.zip_name_for_dataset(dataset)
    zip_path = data.ensure_zip(zname)
    if data.has_qrel_file(zip_path, dataset, "train.tsv"):
        tr_q, tr_r, tr_m = data.load_queries_qrels_file(zip_path, dataset, "train.tsv")
        te_q, te_r, te_m = data.load_queries_qrels_file(zip_path, dataset, "test.tsv")
        return {
            "protocol": "official_train_test",
            "tune_member": tr_m,
            "test_member": te_m,
            "tune_q": tr_q,
            "tune_qrels": tr_r,
            "test_q": te_q,
            "test_qrels": te_r,
        }
    if data.has_qrel_file(zip_path, dataset, "dev.tsv"):
        tr_q, tr_r, tr_m = data.load_queries_qrels_file(zip_path, dataset, "dev.tsv")
        te_q, te_r, te_m = data.load_queries_qrels_file(zip_path, dataset, "test.tsv")
        return {
            "protocol": "official_dev_test",
            "tune_member": tr_m,
            "test_member": te_m,
            "tune_q": tr_q,
            "tune_qrels": tr_r,
            "test_q": te_q,
            "test_qrels": te_r,
        }
    queries, qrels, member = data.load_queries_qrels(zip_path, dataset)
    qids = [str(q["_id"]) for q in queries if str(q["_id"]) in qrels]
    rng = np.random.default_rng(SEED)
    rng.shuffle(qids)
    frac = 0.5 if len(qids) < 40 else 0.7
    n_tune = max(1, min(len(qids) - 1, int(round(len(qids) * frac))))
    tune_ids = set(qids[:n_tune])
    test_ids = set(qids[n_tune:])
    return {
        "protocol": f"heldout_test_seed{SEED}_tune{frac:.1f}",
        "tune_member": member,
        "test_member": member,
        "tune_q": [q for q in queries if str(q["_id"]) in tune_ids],
        "tune_qrels": {k: v for k, v in qrels.items() if k in tune_ids},
        "test_q": [q for q in queries if str(q["_id"]) in test_ids],
        "test_qrels": {k: v for k, v in qrels.items() if k in test_ids},
    }


def _eval_spec(pack, qrels, kind, p):
    return fl.mean_ndcg(
        pack,
        qrels,
        lambda row, k=kind, pp=p: fl.fuse(row["d_ids"], row["d_sc"], row["s_ids"], row["s_sc"], k, **pp),
    )


def _test_metrics(pack, qrels, kind, p) -> dict:
    rows = []
    for row in pack:
        qid = row["qid"]
        if qid not in qrels:
            continue
        ids = [d for d in fl.fuse(row["d_ids"], row["d_sc"], row["s_ids"], row["s_sc"], kind, **p) if d != qid]
        rel = qrels[qid]
        rows.append(
            {
                "ndcg@10": metrics.ndcg_at_k(ids, rel, 10),
                "ndcg@100": metrics.ndcg_at_k(ids, rel, 100),
                "recall@10": metrics.recall_at_k(ids, rel, 10),
                "recall@100": metrics.recall_at_k(ids, rel, 100),
                "mrr@10": metrics.mrr_at_k(ids, rel, 10),
                "map@100": metrics.ap_at_k(ids, rel, 100),
            }
        )
    agg = metrics.aggregate(rows)
    return {k: round(v, 4) for k, v in agg.items()}


def _oracle_alpha(pack, qrels) -> float:
    alphas = [i / 10 for i in range(11)]
    vals = []
    for row in pack:
        if row["qid"] not in qrels:
            continue
        best = -1.0
        for a in alphas:
            ids = fl.fuse(row["d_ids"], row["d_sc"], row["s_ids"], row["s_sc"], "minmax_linear", a=a)
            best = max(best, fl._ndcg10(ids, row["qid"], qrels))
        vals.append(best)
    return round(float(sum(vals) / len(vals)), 4) if vals else 0.0


def _load_state() -> dict:
    if PATH.exists():
        return json.loads(PATH.read_text())
    return {"datasets": {}, "notes": "Per-dataset train-then-freeze of the parameterized hybrid mix."}


def _save(state: dict) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(state, indent=2))


def run_one(dataset: str, c) -> dict:
    t0 = time.perf_counter()
    split = load_tune_test(dataset)
    col = config.hybrid_collection(dataset)
    print(
        f"\n=== {dataset} protocol={split['protocol']} "
        f"tune={len(split['tune_q'])} test={len(split['test_q'])} col={col} ==="
    )
    print("Embedding queries...")
    tr_d, tr_s = embeddings.cached_query_embeddings([q["text"] for q in split["tune_q"]])
    te_d, te_s = embeddings.cached_query_embeddings([q["text"] for q in split["test_q"]])
    print("Retrieving dense + BM42 lists...")
    train_pack = fl.retrieve_pack(split["tune_q"], tr_d, tr_s, c, col)
    test_pack = fl.retrieve_pack(split["test_q"], te_d, te_s, c, col)

    train_scores = []
    for kind, p in SPECS:
        sc = _eval_spec(train_pack, split["tune_qrels"], kind, p)
        train_scores.append({"name": _spec_name(kind, p), "kind": kind, "params": p, "tune_ndcg@10": round(sc, 4)})
    train_scores.sort(key=lambda r: r["tune_ndcg@10"], reverse=True)
    best = train_scores[0]

    test_fixed = {}
    for name, kind, p in [
        ("dense", "dense", {}),
        ("bm42", "sparse", {}),
        ("rrf_equal", "rrf", {"a": 0.5, "kappa": 60}),
        ("tuned", best["kind"], best["params"]),
    ]:
        test_fixed[name] = _test_metrics(test_pack, split["test_qrels"], kind, p)

    # Full α curve on test for the paper (minmax_linear).
    alpha_curve = []
    for a in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        alpha_curve.append(
            {"a": a, "ndcg@10": _test_metrics(test_pack, split["test_qrels"], "minmax_linear", {"a": a})["ndcg@10"]}
        )

    oracle = _oracle_alpha(test_pack, split["test_qrels"])
    dense_n = test_fixed["dense"]["ndcg@10"]
    rrf_n = test_fixed["rrf_equal"]["ndcg@10"]
    tun_n = test_fixed["tuned"]["ndcg@10"]
    out = {
        "dataset": dataset,
        "collection": col,
        "protocol": split["protocol"],
        "n_tune": len(train_pack),
        "n_test": len(test_pack),
        "tune_top": train_scores[:8],
        "winner": best,
        "test": test_fixed,
        "alpha_curve_minmax": alpha_curve,
        "oracle_minmax_alpha_ndcg@10": oracle,
        "beats_dense": tun_n > dense_n,
        "beats_rrf": tun_n > rrf_n,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    print(
        f"  winner {best['name']} tune={best['tune_ndcg@10']} "
        f"test={tun_n} dense={dense_n} rrf={rrf_n} oracle={oracle}"
    )
    return out


def evaluated_datasets() -> list[str]:
    suite = json.loads((config.RESULTS_DIR / "beir_suite.json").read_text())
    names = []
    for r in suite.get("records") or []:
        if r.get("status") == "evaluated" and r.get("dataset"):
            names.append(r["dataset"])
    # Unique, stable order: small zips first, then CQA.
    seen = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return seen


def run_fusion_all(only: list[str] | None = None) -> dict:
    c = qdrant_utils.wait_ready()
    state = _load_state()
    targets = only or evaluated_datasets()
    for ds in targets:
        if ds in state["datasets"] and state["datasets"][ds].get("test"):
            print(f"Skip {ds} (already in fusion_all.json)")
            continue
        try:
            state["datasets"][ds] = run_one(ds, c)
        except Exception as exc:
            print(f"FAIL {ds}: {exc}")
            state["datasets"][ds] = {"dataset": ds, "error": str(exc)}
        _save(state)

    rows = [v for v in state["datasets"].values() if v.get("test")]
    n_beat_d = sum(1 for r in rows if r.get("beats_dense"))
    n_beat_r = sum(1 for r in rows if r.get("beats_rrf"))
    state["summary"] = {
        "n_datasets": len(rows),
        "n_beats_dense": n_beat_d,
        "n_beats_rrf": n_beat_r,
        "mean_test_ndcg@10": {
            "dense": round(sum(r["test"]["dense"]["ndcg@10"] for r in rows) / len(rows), 4) if rows else 0,
            "rrf": round(sum(r["test"]["rrf_equal"]["ndcg@10"] for r in rows) / len(rows), 4) if rows else 0,
            "tuned": round(sum(r["test"]["tuned"]["ndcg@10"] for r in rows) / len(rows), 4) if rows else 0,
        },
    }
    _save(state)
    print("SUMMARY", state["summary"])
    return state


if __name__ == "__main__":
    run_fusion_all()

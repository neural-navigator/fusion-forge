from __future__ import annotations

import json
import shutil
from typing import Any

from qdrant_client.http import models

from . import config, data, embeddings, metrics, qdrant_utils, search
from .index import index_hybrid_stream


def _free_gb() -> float:
    return shutil.disk_usage("/").free / (1024**3)


def _suite_path():
    return config.RESULTS_DIR / "beir_suite.json"


def load_records() -> list[dict[str, Any]]:
    path = _suite_path()
    backup = config.RESULTS_DIR / "beir_suite_small.json"
    by_ds: dict[str, dict[str, Any]] = {}
    for p in (backup, path):
        if not p.exists():
            continue
        try:
            raw = json.loads(p.read_text())
        except Exception:
            continue
        recs = raw.get("records", raw if isinstance(raw, list) else [])
        for r in recs:
            if isinstance(r, dict) and r.get("dataset"):
                by_ds[r["dataset"]] = r
    return list(by_ds.values())


def evaluate_hybrid_collection(
    dataset: str, queries: list[dict], qrels: dict[str, dict[str, int]]
) -> dict[str, Any]:
    c = qdrant_utils.wait_ready()
    col = config.hybrid_collection(dataset)
    texts = [q["text"] for q in queries]
    q_dense, q_sparse = embeddings.cached_query_embeddings(texts)
    qid_to_i = {str(q["_id"]): i for i, q in enumerate(queries)}

    def dense_ret(qid: str, text: str):
        i = qid_to_i[qid]
        ids, _, dt = search.search_dense(
            c, col, q_dense[i].tolist(), using="dense"
        )
        return ids, dt

    def bm42_ret(qid: str, text: str):
        i = qid_to_i[qid]
        ids, _, dt = search.search_sparse(c, col, q_sparse[i], using="bm42")
        return ids, dt

    def hybrid_ret(qid: str, text: str):
        i = qid_to_i[qid]
        ids, _, dt = search.search_fusion(
            c, col, q_dense[i].tolist(), q_sparse[i], models.Fusion.RRF
        )
        return ids, dt

    runs = [
        metrics.evaluate_run("dense_bge_base", dense_ret, queries, qrels, extra={"using": "dense"}),
        metrics.evaluate_run("bm42_idf", bm42_ret, queries, qrels, extra={"using": "bm42"}),
        metrics.evaluate_run("hybrid_rrf", hybrid_ret, queries, qrels, extra={"fusion": "RRF"}),
    ]
    return {
        "dataset": dataset,
        "n_queries": len(queries),
        "collection": col,
        "runs": runs,
    }


def _save_all(records: list[dict]) -> dict[str, Any]:
    by_ds: dict[str, dict] = {}
    for r in records:
        if r.get("dataset"):
            by_ds[r["dataset"]] = r
    merged = list(by_ds.values())
    out = finalize_payload(merged)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _suite_path().write_text(json.dumps(out, indent=2, default=str))
    return out


def cqadupstack_macro(records: list[dict]) -> dict[str, Any] | None:
    subs = [
        r
        for r in records
        if str(r.get("dataset", "")).startswith("cqadupstack/")
        and r.get("status") == "evaluated"
        and r.get("eval")
    ]
    if not subs:
        return None
    keys = ["ndcg@10", "ndcg@100", "recall@10", "recall@100", "precision@10", "mrr@10", "map@100"]
    run_names = ["dense_bge_base", "bm42_idf", "hybrid_rrf"]
    avg_runs = []
    for name in run_names:
        mets = {k: [] for k in keys}
        lats = {"mean": [], "p50": [], "p95": []}
        for r in subs:
            run = next((x for x in r["eval"]["runs"] if x["name"] == name), None)
            if not run:
                continue
            for k in keys:
                mets[k].append(run["metrics"][k])
            for k in lats:
                lats[k].append(run["latency_ms"][k])
        avg_runs.append(
            {
                "name": name,
                "n_queries": "macro",
                "metrics": {k: round(sum(v) / len(v), 4) for k, v in mets.items() if v},
                "latency_ms": {k: round(sum(v) / len(v), 2) for k, v in lats.items() if v},
                "extra": {"aggregate": "macro-average over CQADupStack subcollections"},
            }
        )
    return {
        "dataset": "cqadupstack (macro-avg)",
        "n_docs": sum(r.get("n_docs") or 0 for r in subs),
        "n_eval_queries": sum(r.get("n_eval_queries") or 0 for r in subs),
        "n_subcollections": len(subs),
        "eval": {
            "dataset": "cqadupstack",
            "n_queries": sum(r["eval"]["n_queries"] for r in subs),
            "collection": "per-subcollection",
            "runs": avg_runs,
        },
        "status": "evaluated",
        "aggregate": True,
    }


def _summary_rows(records: list[dict]) -> list[dict]:
    rows = []
    for r in records:
        if r.get("status") != "evaluated" or not r.get("eval"):
            continue
        metrics_by = {run["name"]: run["metrics"] for run in r["eval"]["runs"]}
        rows.append(
            {
                "dataset": r["dataset"],
                "n_docs": r.get("n_docs"),
                "n_queries": r["eval"].get("n_queries"),
                "dense_ndcg@10": metrics_by.get("dense_bge_base", {}).get("ndcg@10"),
                "bm42_ndcg@10": metrics_by.get("bm42_idf", {}).get("ndcg@10"),
                "hybrid_ndcg@10": metrics_by.get("hybrid_rrf", {}).get("ndcg@10"),
            }
        )
    return rows


def finalize_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    license_rows = [
        {
            "zip_name": name,
            "dataset": name,
            "status": "skipped_license",
            "skip_reason": reason,
            "n_docs": None,
            "zip_gb": None,
        }
        for name, reason in config.BEIR_LICENSE_MISSING
    ]
    by_ds = {r["dataset"]: r for r in records if r.get("dataset")}
    for row in license_rows:
        by_ds.setdefault(row["dataset"], row)
    merged = list(by_ds.values())
    cqa = cqadupstack_macro(merged)
    evaluated = [r for r in merged if r.get("status") == "evaluated"]
    skipped = [r for r in merged if r.get("status") != "evaluated"]
    unit_summary = _summary_rows(evaluated)
    cqa_summary = _summary_rows([cqa]) if cqa else []
    zip_level = [
        r
        for r in unit_summary
        if not str(r["dataset"]).startswith("cqadupstack/")
    ] + cqa_summary
    return {
        "n_evaluated": len(evaluated),
        "n_skipped": len(skipped),
        "summary": unit_summary + cqa_summary,
        "summary_zip_level": zip_level,
        "cqadupstack_macro": cqa,
        "records": merged,
        "models": {"dense": config.DENSE_MODEL, "sparse": config.SPARSE_MODEL},
        "limits": {
            "max_index_docs": config.MAX_INDEX_DOCS,
            "min_free_gb": config.MIN_FREE_GB_TO_INDEX,
        },
        "license_missing": [n for n, _ in config.BEIR_LICENSE_MISSING],
    }


def process_zip(
    zip_name: str,
    drop_zip_after: bool = False,
    skip_datasets: set[str] | None = None,
) -> list[dict[str, Any]]:
    skip_datasets = skip_datasets or set()
    dest = config.DATA_DIR / f"{zip_name}.zip"
    if not dest.exists() and zip_name in config.WIKI_ZIPS:
        remote = data.remote_zip_bytes(zip_name)
        need_gb = (remote / (1024**3)) if remote else 8.0
        free = _free_gb()
        rec = {
            "zip_name": zip_name,
            "dataset": zip_name,
            "status": "skipped_hardware",
            "skip_reason": (
                f"wiki-scale BEIR dump ({need_gb:.2f} GB zip"
                f"{'' if remote else ' estimate'}); {free:.1f} GB free. "
                f"Corpus is 2M–8.8M docs, above MAX_INDEX_DOCS={config.MAX_INDEX_DOCS} "
                "for a 768-d hybrid index on this workstation."
            ),
            "zip_gb": round(need_gb, 3) if remote else None,
            "n_docs": None,
            "free_gb_before": round(free, 2),
        }
        print("SKIP wiki", rec["skip_reason"])
        return [rec]
    if not dest.exists():
        remote = data.remote_zip_bytes(zip_name)
        if remote is not None:
            need_gb = remote / (1024**3)
            free = _free_gb()
            if need_gb + 1.5 > free:
                rec = {
                    "zip_name": zip_name,
                    "dataset": zip_name,
                    "status": "skipped_hardware",
                    "skip_reason": (
                        f"remote zip is {need_gb:.2f} GB; only {free:.1f} GB free "
                        "(need ~1.5 GB headroom to download)"
                    ),
                    "zip_gb": round(need_gb, 3),
                    "n_docs": None,
                    "free_gb_before": round(free, 2),
                }
                print("SKIP download", rec["skip_reason"])
                return [rec]
    zip_path = data.ensure_zip(zip_name)
    zip_gb = zip_path.stat().st_size / 1e9
    units = data.dataset_units(zip_name, zip_path)
    records = []
    for unit in units:
        ds = unit["dataset"]
        if ds in skip_datasets:
            print(f"Resume skip {ds} (already evaluated)")
            continue
        print(f"\n=== {ds} (zip {zip_name}, corpus {unit['corpus_bytes'] / 1e6:.1f} MB uncompressed) ===")
        # Avoid a full zip scan on multi-GB archives; ~1.8KB/doc is conservative for BEIR.
        n_docs = max(1, int(unit["corpus_bytes"] / 1800))
        if unit["corpus_bytes"] <= 8_000_000:
            n_docs = data.count_corpus_lines(zip_path, unit["corpus_member"])
        rec: dict[str, Any] = {
            **unit,
            "zip_gb": round(zip_gb, 3),
            "n_docs": n_docs,
            "free_gb_before": round(_free_gb(), 2),
        }
        estimated_index_gb = n_docs * 768 * 4 / 1e9 * 1.4
        skip_reason = None
        if n_docs > config.MAX_INDEX_DOCS:
            skip_reason = (
                f"corpus has {n_docs} docs (> {config.MAX_INDEX_DOCS}); "
                f"768-d hybrid would need ~{estimated_index_gb:.1f} GB vectors plus payload"
            )
        elif _free_gb() < config.MIN_FREE_GB_TO_INDEX:
            skip_reason = f"only {_free_gb():.1f} GB free (< {config.MIN_FREE_GB_TO_INDEX})"
        elif estimated_index_gb > _free_gb() - 2:
            skip_reason = (
                f"estimated dense storage {estimated_index_gb:.1f} GB exceeds free disk "
                f"{_free_gb():.1f} GB"
            )
        if skip_reason:
            rec["status"] = "skipped_hardware"
            rec["skip_reason"] = skip_reason
            print("SKIP", skip_reason)
            records.append(rec)
            _save_all(load_records() + records)
            continue
        col = config.hybrid_collection(ds)
        c = qdrant_utils.wait_ready()
        try:
            queries, qrels, qrel_member = data.load_queries_qrels(zip_path, ds)
        except Exception as exc:
            rec["status"] = "skipped_qrels"
            rec["skip_reason"] = str(exc)
            records.append(rec)
            continue
        rec["n_eval_queries"] = len(queries)
        rec["qrel_member"] = qrel_member
        if c.collection_exists(col):
            existing = c.get_collection(col).points_count or 0
            if existing >= max(int(n_docs * 0.9), 1):
                print(f"Reusing existing {col}")
                rec["index"] = {
                    "collection": col,
                    "reused": True,
                    "resources": qdrant_utils.collection_resources(c, col),
                }
            else:
                print(f"Partial {col} ({existing} pts), reindexing")
                store_text = n_docs <= 20_000
                rec["index"] = index_hybrid_stream(
                    ds, zip_path, unit["corpus_member"], n_docs, store_text=store_text
                )
        else:
            store_text = n_docs <= 20_000
            rec["index"] = index_hybrid_stream(
                ds, zip_path, unit["corpus_member"], n_docs, store_text=store_text
            )
        rec["eval"] = evaluate_hybrid_collection(ds, queries, qrels)
        rec["status"] = "evaluated"
        rec["free_gb_after"] = round(_free_gb(), 2)
        records.append(rec)
        _save_all(load_records() + records)
    if drop_zip_after and zip_path.exists() and zip_gb > 0.4:
        print(f"Removing {zip_path} to free disk")
        zip_path.unlink()
    return records


def run_beir_suite() -> dict[str, Any]:
    c = qdrant_utils.wait_ready()
    dropped = qdrant_utils.drop_extra_scifact_collections(c)
    print("Dropped extra SciFact collections:", dropped)
    all_records = load_records()
    done = {r["dataset"] for r in all_records if r.get("status") == "evaluated"}
    print(f"Resuming with {len(done)} already-evaluated units")
    for name in config.SMALL_ZIPS:
        if name in done:
            print(f"Resume skip zip {name}")
            continue
        all_records.extend(process_zip(name, drop_zip_after=False, skip_datasets=done))
        done = {r["dataset"] for r in all_records if r.get("status") == "evaluated"}
        _save_all(all_records)
    all_records.extend(process_zip("cqadupstack", drop_zip_after=True, skip_datasets=done))
    done = {r["dataset"] for r in all_records if r.get("status") == "evaluated"}
    _save_all(all_records)
    for name in config.WIKI_ZIPS:
        already = any(
            r.get("zip_name") == name or r.get("dataset") == name for r in all_records
        )
        if already and any(
            r.get("dataset") == name and r.get("status") != "evaluated" for r in all_records
        ):
            pass
        all_records.extend(process_zip(name, drop_zip_after=True, skip_datasets=done))
        done = {r["dataset"] for r in all_records if r.get("status") == "evaluated"}
        _save_all(all_records)

    out = _save_all(all_records)
    out["dropped_collections"] = dropped
    _suite_path().write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote beir_suite.json ({out['n_evaluated']} evaluated, {out['n_skipped']} skipped)")
    return out


if __name__ == "__main__":
    run_beir_suite()

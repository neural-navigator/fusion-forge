"""Finish remaining CQADupStack units from compact extracted files (no zip)."""
from __future__ import annotations

import json
from pathlib import Path

from qdrant_client.http import models
from tqdm import tqdm

from . import config, data, embeddings, metrics, qdrant_utils, search
from .beir_suite import _save_all, load_records
from .index import _flush


EXTRACT = config.DATA_DIR / "extract"
REMAINING = ["tex", "unix", "webmasters", "wordpress"]


def _load_eval(sub: str):
    queries = []
    qp = EXTRACT / f"cqadupstack_{sub}_queries_eval.jsonl"
    with qp.open() as fh:
        for line in fh:
            if line.strip():
                queries.append(json.loads(line))
    qrels: dict[str, dict[str, int]] = {}
    with (EXTRACT / f"cqadupstack_{sub}_qrels_test.tsv").open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3 or parts[0] in {"query-id", "qid"}:
                continue
            qrels.setdefault(parts[0], {})[parts[1]] = int(parts[-1])
    qids = set(qrels)
    queries = [q for q in queries if str(q["_id"]) in qids]
    return queries, qrels


def _index_file(dataset: str, corpus_path: Path, n_docs: int) -> dict:
    c = qdrant_utils.wait_ready()
    name = config.hybrid_collection(dataset)
    qdrant_utils.recreate_hybrid(c, name, on_disk=True)
    pid = 0
    batch: list[dict] = []
    import time

    t0 = time.perf_counter()
    with corpus_path.open() as fh:
        pbar = tqdm(fh, total=n_docs, desc=f"index {dataset}", unit="doc")
        for line in pbar:
            if not line.strip():
                continue
            batch.append(json.loads(line))
            if len(batch) >= config.UPSERT_BATCH:
                _flush(c, name, batch, pid, store_text=False)
                pid += len(batch)
                batch = []
        if batch:
            _flush(c, name, batch, pid, store_text=False)
            pid += len(batch)
    resources = qdrant_utils.collection_resources(c, name)
    print(f"Indexed {name}: {time.perf_counter()-t0:.1f}s, {resources.get('points_count')} pts")
    return {"collection": name, "upserted": pid, "resources": resources}


def _eval(dataset: str, queries, qrels) -> dict:
    c = qdrant_utils.wait_ready()
    col = config.hybrid_collection(dataset)
    texts = [q["text"] for q in queries]
    q_dense, q_sparse = embeddings.cached_query_embeddings(texts)
    qid_to_i = {str(q["_id"]): i for i, q in enumerate(queries)}

    def dense_ret(qid, text):
        i = qid_to_i[qid]
        ids, _, dt = search.search_dense(c, col, q_dense[i].tolist(), using="dense")
        return ids, dt

    def bm42_ret(qid, text):
        i = qid_to_i[qid]
        ids, _, dt = search.search_sparse(c, col, q_sparse[i], using="bm42")
        return ids, dt

    def hybrid_ret(qid, text):
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
    return {"dataset": dataset, "n_queries": len(queries), "collection": col, "runs": runs}


def main() -> None:
    records = load_records()
    done = {r["dataset"] for r in records if r.get("status") == "evaluated"}
    for sub in REMAINING:
        ds = f"cqadupstack/{sub}"
        if ds in done:
            print("skip", ds)
            continue
        corpus = EXTRACT / f"cqadupstack_{sub}_corpus.jsonl"
        n_docs = sum(1 for _ in corpus.open())
        print(f"=== {ds} n_docs={n_docs} ===")
        queries, qrels = _load_eval(sub)
        rec = {
            "zip_name": "cqadupstack",
            "dataset": ds,
            "n_docs": n_docs,
            "n_eval_queries": len(queries),
            "index": _index_file(ds, corpus, n_docs),
            "eval": _eval(ds, queries, qrels),
            "status": "evaluated",
        }
        records.append(rec)
        _save_all(records)
        print("saved", ds)

    # wiki skips
    from .beir_suite import process_zip

    done = {r["dataset"] for r in load_records() if r.get("status") == "evaluated"}
    extra = []
    for name in config.WIKI_ZIPS:
        if name in done:
            continue
        extra.extend(process_zip(name, drop_zip_after=True, skip_datasets=done))
    if extra:
        _save_all(load_records() + extra)
    print("remaining CQA + wiki skips done")


if __name__ == "__main__":
    main()

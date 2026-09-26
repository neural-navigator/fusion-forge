"""Re-evaluate existing Qdrant collections after eval-protocol fixes. No reindex."""
from __future__ import annotations

import json
from pathlib import Path

from src import config, data, qdrant_utils
from src.beir_suite import _save_all, evaluate_hybrid_collection, finalize_payload, load_records
from src.experiments import run_baseline, run_quantization, run_scoring
from src.finish_cqa import _load_eval
from src.ranking import run_ranking
from src.report import build_pdf
from src.router import run_router
from src.run_all import snapshot_hardware, _qdrant_version


CQA_EXTRACT = {"tex", "unix", "webmasters", "wordpress"}


def _patch_index_disk(c) -> None:
    path = config.RESULTS_DIR / "index.json"
    if not path.exists():
        return
    meta = json.loads(path.read_text())
    resources = meta.get("resources") or {}
    for name in list(resources):
        if c.collection_exists(name):
            resources[name] = qdrant_utils.collection_resources(c, name)
    meta["resources"] = resources
    meta["qdrant_version"] = _qdrant_version()
    path.write_text(json.dumps(meta, indent=2, default=str))
    print("Updated disk figures in index.json")


def reeval_zip_dataset(ds: str, zip_name: str) -> dict:
    zip_path = config.DATA_DIR / f"{zip_name}.zip"
    queries, qrels, qrel_member = data.load_queries_qrels(zip_path, ds)
    print(f"Eval {ds}: {len(queries)} queries")
    ev = evaluate_hybrid_collection(ds, queries, qrels)
    ev["qrel_member"] = qrel_member
    return ev


def main() -> None:
    c = qdrant_utils.wait_ready()
    records = load_records()
    by_ds = {r["dataset"]: r for r in records if r.get("dataset")}

    for rec in list(by_ds.values()):
        if rec.get("status") != "evaluated":
            continue
        ds = rec["dataset"]
        col = config.hybrid_collection(ds)
        if not c.collection_exists(col):
            print("SKIP missing collection", col)
            continue
        rec["index"] = rec.get("index") or {}
        rec["index"]["resources"] = qdrant_utils.collection_resources(c, col)
        rec["n_docs"] = rec["index"]["resources"].get("points_count") or rec.get("n_docs")

        if ds.startswith("cqadupstack/"):
            sub = ds.split("/", 1)[1]
            if sub in CQA_EXTRACT:
                queries, qrels = _load_eval(sub)
                print(f"Eval {ds} from extract: {len(queries)} queries")
                rec["eval"] = evaluate_hybrid_collection(ds, queries, qrels)
            else:
                print(f"KEEP prior eval for {ds} (no compact queries on disk)")
                continue
        else:
            zip_name = rec.get("zip_name") or ds
            zpath = config.DATA_DIR / f"{zip_name}.zip"
            if not zpath.exists():
                print("SKIP no zip", ds)
                continue
            rec["eval"] = reeval_zip_dataset(ds, zip_name)
            rec["n_eval_queries"] = rec["eval"]["n_queries"]

    out = _save_all(list(by_ds.values()))
    print("Suite", out["n_evaluated"], "evaluated")

    print("=== SciFact extras ===")
    run_scoring()
    run_quantization()
    run_router()
    run_ranking()
    run_baseline()
    _patch_index_disk(c)

    hw = snapshot_hardware()
    hw["qdrant_version"] = _qdrant_version()
    hw["note"] = "Snapshot after re-eval with allocated disk (du -sk) and ignore_identical_ids."
    (config.RESULTS_DIR / "hardware.json").write_text(json.dumps(hw, indent=2))

    pdf = build_pdf()
    print("PDF", pdf)


if __name__ == "__main__":
    main()

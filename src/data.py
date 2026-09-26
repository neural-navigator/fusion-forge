from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from pathlib import Path

import requests

from . import config

QREL_NAMES = ("test.tsv", "test.txt", "dev.tsv", "dev.txt")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    tmp.replace(dest)


def zip_url(name: str) -> str:
    return f"{config.BEIR_BASE_URL}/{name}.zip"


def remote_zip_bytes(name: str) -> int | None:
    try:
        resp = requests.head(zip_url(name), timeout=60, allow_redirects=True)
        resp.raise_for_status()
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl else None
    except Exception:
        return None


def ensure_zip(name: str) -> Path:
    dest = config.DATA_DIR / f"{name}.zip"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"Downloading {name}.zip")
    _download(zip_url(name), dest)
    print(f"  saved {dest.stat().st_size / 1e9:.2f} GB")
    return dest


def zip_members(zip_path: Path, suffix: str) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(zip_path) as zf:
        return [
            info
            for info in zf.infolist()
            if info.filename.endswith(suffix) and not info.is_dir()
        ]


def dataset_units(zip_name: str, zip_path: Path) -> list[dict]:
    """One unit per corpus.jsonl (CQADupStack has 12)."""
    corpora = zip_members(zip_path, "corpus.jsonl")
    units = []
    for info in corpora:
        parent = str(Path(info.filename).parent).replace("\\", "/")
        parts = [p for p in parent.split("/") if p]
        if zip_name == "cqadupstack" and len(parts) >= 2:
            ds_id = f"cqadupstack/{parts[-1]}"
        elif parts:
            ds_id = parts[-1]
        else:
            ds_id = zip_name
        units.append(
            {
                "zip_name": zip_name,
                "dataset": ds_id,
                "corpus_member": info.filename,
                "corpus_bytes": info.file_size,
                "corpus_compress_bytes": info.compress_size,
            }
        )
    units.sort(key=lambda u: u["dataset"])
    return units


def count_corpus_lines(zip_path: Path, member: str) -> int:
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            for _ in fh:
                n += 1
    return n


def iter_jsonl_file(path: Path) -> Iterator[dict]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_zip_member(zip_path: Path, member: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"Extracting {member} -> {dest}")
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as src, dest.open("wb") as out:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    return dest


def iter_jsonl_from_zip(zip_path: Path, member: str) -> Iterator[dict]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            for raw in fh:
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    yield json.loads(line)


def _find_member(zf: zipfile.ZipFile, dataset: str, filename: str) -> str | None:
    candidates = []
    for name in zf.namelist():
        if name.endswith("/" + filename) or name.endswith(filename):
            if dataset.startswith("cqadupstack/"):
                sub = dataset.split("/", 1)[1]
                if f"/{sub}/" in "/" + name.replace("\\", "/"):
                    candidates.append(name)
            else:
                candidates.append(name)
    if not candidates:
        return None
    candidates.sort(key=len)
    return candidates[0]


def load_queries_qrels(
    zip_path: Path, dataset: str
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    with zipfile.ZipFile(zip_path) as zf:
        qrel_member = None
        for fname in QREL_NAMES:
            qrel_member = _find_member(zf, dataset, fname)
            if qrel_member and "/qrels/" in qrel_member.replace("\\", "/"):
                break
            qrel_member = None
        if qrel_member is None:
            for fname in QREL_NAMES:
                qrel_member = _find_member(zf, dataset, fname)
                if qrel_member:
                    break
        if not qrel_member:
            raise FileNotFoundError(f"qrels missing for {dataset}")
        qrels: dict[str, dict[str, int]] = {}
        with zf.open(qrel_member) as fh:
            for raw in fh:
                parts = raw.decode("utf-8", errors="replace").strip().split()
                if len(parts) < 3 or parts[0] in {"query-id", "qid"}:
                    continue
                if len(parts) >= 4:
                    qid, did, rel = parts[0], parts[2], int(parts[3])
                else:
                    qid, did, rel = parts[0], parts[1], int(parts[-1])
                qrels.setdefault(qid, {})[did] = rel
        eval_qids = set(qrels)
        q_member = _find_member(zf, dataset, "queries.jsonl")
        if not q_member:
            raise FileNotFoundError(f"queries.jsonl missing for {dataset}")
        queries = []
        with zf.open(q_member) as fh:
            for raw in fh:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                obj = json.loads(line)
                qid = str(obj["_id"])
                if qid not in eval_qids:
                    continue
                queries.append({"_id": qid, "text": (obj.get("text") or "")[:8000]})
                if len(queries) >= len(eval_qids):
                    break
                if config.MAX_EVAL_QUERIES and len(queries) >= config.MAX_EVAL_QUERIES:
                    break
    return queries, qrels, qrel_member


def load_queries_qrels_file(
    zip_path: Path, dataset: str, qrel_filename: str
) -> tuple[list[dict], dict[str, dict[str, int]], str]:
    """Load queries whose ids appear in a named qrels file (train.tsv, test.tsv, …)."""
    with zipfile.ZipFile(zip_path) as zf:
        qrel_member = _find_member(zf, dataset, qrel_filename)
        if not qrel_member:
            raise FileNotFoundError(f"{qrel_filename} missing for {dataset}")
        qrels: dict[str, dict[str, int]] = {}
        with zf.open(qrel_member) as fh:
            for raw in fh:
                parts = raw.decode("utf-8", errors="replace").strip().split()
                if len(parts) < 3 or parts[0] in {"query-id", "qid"}:
                    continue
                if len(parts) >= 4:
                    qid, did, rel = parts[0], parts[2], int(parts[3])
                else:
                    qid, did, rel = parts[0], parts[1], int(parts[-1])
                qrels.setdefault(qid, {})[did] = rel
        eval_qids = set(qrels)
        q_member = _find_member(zf, dataset, "queries.jsonl")
        if not q_member:
            raise FileNotFoundError(f"queries.jsonl missing for {dataset}")
        queries = []
        with zf.open(q_member) as fh:
            for raw in fh:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                obj = json.loads(line)
                qid = str(obj["_id"])
                if qid not in eval_qids:
                    continue
                queries.append({"_id": qid, "text": (obj.get("text") or "")[:8000]})
                if len(queries) >= len(eval_qids):
                    break
    return queries, qrels, qrel_member


def has_qrel_file(zip_path: Path, dataset: str, qrel_filename: str) -> bool:
    with zipfile.ZipFile(zip_path) as zf:
        return _find_member(zf, dataset, qrel_filename) is not None


def zip_name_for_dataset(dataset: str) -> str:
    if dataset.startswith("cqadupstack/"):
        return "cqadupstack"
    return dataset


def load_qrels_tsv(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3 or parts[0] in {"query-id", "qid"}:
                continue
            if len(parts) >= 4:
                qid, did, rel = parts[0], parts[2], int(parts[3])
            else:
                qid, did, rel = parts[0], parts[1], int(parts[-1])
            qrels.setdefault(qid, {})[did] = rel
    return qrels


def load_queries_jsonl(path: Path, qids: set[str] | None = None) -> list[dict]:
    queries = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj["_id"])
            if qids is not None and qid not in qids:
                continue
            queries.append({"_id": qid, "text": (obj.get("text") or "")[:8000]})
    return queries


def load_scifact_split(split: str) -> tuple[list[dict], dict[str, dict[str, int]]]:
    qrels = load_qrels_tsv(config.DATA_DIR / "scifact" / "qrels" / f"{split}.tsv")
    qpath = config.DATA_DIR / "scifact" / "queries.jsonl"
    queries = load_queries_jsonl(qpath, set(qrels))
    return queries, qrels


def load_dataset(name: str | None = None):
    """Deep-dive helper: materialize SciFact (or named zip) into memory."""
    name = name or config.DEEP_DIVE_DATASET
    zip_path = ensure_zip(name if name in config.BEIR_ZIPS else "scifact")
    units = [u for u in dataset_units("scifact" if name == "scifact" else name, zip_path) if u["dataset"] == name or (name == "scifact" and u["dataset"] == "scifact")]
    if not units:
        units = dataset_units(name, zip_path)
    unit = units[0]
    corpus = list(iter_jsonl_from_zip(zip_path, unit["corpus_member"]))
    queries, qrels, _ = load_queries_qrels(zip_path, unit["dataset"])
    return corpus, queries, qrels


def doc_text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    if title and text:
        out = f"{title}. {text}"
    else:
        out = title or text
    return out[:1500]

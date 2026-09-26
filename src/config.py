from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
CACHE_DIR = RESULTS_DIR / "cache"

QDRANT_URL = "http://127.0.0.1:6333"
BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

DENSE_MODEL = "BAAI/bge-base-en-v1.5"
DENSE_DIM = 768
SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Official English BEIR dumps on the UKP mirror (excludes mMARCO/Mr.TyDi/MSMARCO-v2/GermanQuAD).
BEIR_ZIPS = [
    "arguana",
    "climate-fever",
    "cqadupstack",
    "dbpedia-entity",
    "fever",
    "fiqa",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quora",
    "scidocs",
    "scifact",
    "trec-covid",
    "webis-touche2020",
]

# Present in the BEIR paper but not on the public UKP zip mirror (dataset licenses).
BEIR_LICENSE_MISSING = [
    ("bioasq", "Not on the public UKP BEIR zip mirror (BioASQ license)."),
    ("signal1m", "Not on the public UKP BEIR zip mirror (Signal-1M license)."),
    ("trec-news", "Not on the public UKP BEIR zip mirror (TREC-NEWS license)."),
    ("robust04", "Not on the public UKP BEIR zip mirror (Robust04/TREC license)."),
]

SMALL_ZIPS = [
    "nfcorpus",
    "scifact",
    "arguana",
    "fiqa",
    "scidocs",
    "trec-covid",
    "quora",
    "webis-touche2020",
]
WIKI_ZIPS = ["nq", "dbpedia-entity", "hotpotqa", "fever", "climate-fever", "msmarco"]

# Index if corpus docs <= this. Wiki-scale BEIR sets exceed workstation disk/RAM at 768-d.
MAX_INDEX_DOCS = 800_000
MIN_FREE_GB_TO_INDEX = 2.0
DEEP_DIVE_DATASET = "scifact"

COL_DENSE = "scifact_dense"
COL_BM42 = "scifact_bm42"
COL_HYBRID = "beir_scifact_hyb"
COL_DENSE_DOT = "scifact_dense_dot"
COL_DENSE_EUCLID = "scifact_dense_euclid"
COL_BM42_NO_IDF = "scifact_bm42_no_idf"
COL_Q_SCALAR = "scifact_dense_scalar"
COL_Q_BINARY = "scifact_dense_binary"
COL_Q_PQ = "scifact_dense_pq"

UPSERT_BATCH = 32
EMBED_BATCH = 32
QUERY_K = 100
METRIC_KS = (10, 100)
PREFETCH_LIMIT = QUERY_K
RERANK_CANDIDATES = 50
MAX_EVAL_QUERIES = None
DATASET = DEEP_DIVE_DATASET

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def hybrid_collection(dataset: str) -> str:
    safe = dataset.replace("/", "_").replace("-", "_")
    return f"beir_{safe}_hyb"

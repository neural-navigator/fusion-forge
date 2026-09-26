from __future__ import annotations

import numpy as np
from tqdm import tqdm

from . import config


class Embedders:
    def __init__(self) -> None:
        self._dense = None
        self._sparse = None

    @property
    def dense(self):
        if self._dense is None:
            from sentence_transformers import SentenceTransformer

            device = "cpu"
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
            except Exception:
                device = "cpu"
            print(f"Loading dense model {config.DENSE_MODEL} on {device}")
            self._dense = SentenceTransformer(config.DENSE_MODEL, device=device)
        return self._dense

    @property
    def sparse(self):
        if self._sparse is None:
            print(f"Loading BM42 model {config.SPARSE_MODEL}")
            from fastembed import SparseTextEmbedding

            self._sparse = SparseTextEmbedding(
                model_name=config.SPARSE_MODEL, threads=4
            )
        return self._sparse


EMBEDDERS = Embedders()


def embed_dense(texts: list[str], query: bool = False) -> np.ndarray:
    if query:
        texts = [config.BGE_QUERY_PREFIX + t for t in texts]
    vecs = EMBEDDERS.dense.encode(
        texts,
        batch_size=min(config.EMBED_BATCH, max(len(texts), 1)),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 64,
    )
    return np.asarray(vecs, dtype=np.float32)


def embed_sparse(texts: list[str]) -> list[dict]:
    out = []
    for vec in EMBEDDERS.sparse.embed(texts, batch_size=config.EMBED_BATCH):
        out.append(
            {
                "indices": vec.indices.astype(int).tolist(),
                "values": vec.values.astype(float).tolist(),
            }
        )
    return out


def cached_query_embeddings(query_texts: list[str]):
    """Queries are small; embed in RAM without writing giant cache files."""
    return embed_dense(query_texts, query=True), embed_sparse(query_texts)

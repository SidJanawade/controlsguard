"""Hybrid retrieval: BM25 keyword search + dense vector search, fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .embeddings import Embedder, TfidfSvdEmbedder
from .ingestion import Chunk
from .textutil import tokenize


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs_tokens]
        self.dl = np.array([len(d) for d in docs_tokens], dtype=float)
        self.avgdl = float(self.dl.mean()) if len(docs_tokens) else 0.0
        df: Counter = Counter()
        for d in self.tf:
            df.update(d.keys())
        n = len(docs_tokens)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / (self.avgdl or 1))
                    s[i] += idf * f * (self.k1 + 1) / denom
        return s


class InMemoryVectorStore:
    """Cosine search over a normalised matrix. Swap for pgvector / Chroma / OpenSearch in production."""

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    def search(self, qvec: np.ndarray) -> np.ndarray:
        return self.matrix @ qvec


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict = field(default_factory=dict)  # which retriever ranked it where (for explainability)


class HybridRetriever:
    RRF_K = 60

    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        self.chunks = chunks
        self._tokens = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25(self._tokens)
        self.embedder = (embedder or TfidfSvdEmbedder()).fit([c.text for c in chunks])
        self.store = InMemoryVectorStore(self.embedder.embed([c.text for c in chunks]))
        self.vocab = set(self.bm25.idf)
        self._by_entity = {c.entity_id.upper(): c for c in chunks if c.entity_id}

    # -- lookups -----------------------------------------------------------------------------
    def by_entity(self, entity_id: str) -> Chunk | None:
        return self._by_entity.get(entity_id.upper())

    def query_coverage(self, query: str) -> float:
        """Fraction of the query's content words that exist anywhere in the corpus (off-topic detector)."""
        toks = [t for t in tokenize(query) if len(t) > 1]
        if not toks:
            return 0.0
        return sum(t in self.vocab for t in toks) / len(toks)

    # -- search ------------------------------------------------------------------------------
    def search(self, query: str, k: int = 4, mode: str = "hybrid", doc_id: str | None = None) -> list[Hit]:
        qtok = tokenize(query)
        bm = self.bm25.scores(qtok)
        vec = self.store.search(self.embedder.embed([query])[0])
        allowed = [i for i, c in enumerate(self.chunks) if doc_id is None or c.doc_id == doc_id]

        def ranking(scores: np.ndarray, require_positive: bool) -> list[int]:
            order = sorted(allowed, key=lambda i: -scores[i])
            return [i for i in order if (scores[i] > 0 or not require_positive)]

        bm_rank = ranking(bm, True)
        vec_rank = ranking(vec, False)
        if mode == "bm25":
            fused = {i: bm[i] for i in bm_rank}
        elif mode == "vector":
            fused = {i: float(vec[i]) for i in vec_rank}
        else:
            fused: dict[int, float] = {}
            for rank_list in (bm_rank, vec_rank):
                for r, i in enumerate(rank_list):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (self.RRF_K + r + 1)
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [
            Hit(
                chunk=self.chunks[i],
                score=float(fused[i]),
                ranks={
                    "bm25": bm_rank.index(i) + 1 if i in bm_rank else None,
                    "vector": vec_rank.index(i) + 1 if i in vec_rank else None,
                },
            )
            for i in top
        ]

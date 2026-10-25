"""Embedding interface plus an offline implementation.

`TfidfSvdEmbedder` (latent semantic analysis) needs no network or API key, so the project runs and is
testable anywhere. For production, implement the same two-method `Embedder` protocol on top of a hosted
embedding endpoint (OpenAI / Gemini / Bedrock) and pass it to `HybridRetriever`; nothing else changes.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .textutil import tokenize


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> "Embedder": ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


class TfidfSvdEmbedder:
    def __init__(self, n_components: int = 64):
        self.n_components = n_components
        self._tfidf = TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True
        )
        self._svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> "TfidfSvdEmbedder":
        X = self._tfidf.fit_transform(texts)
        k = max(2, min(self.n_components, X.shape[0] - 1, X.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=k, random_state=7).fit(X)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        assert self._svd is not None, "call fit() first"
        Z = self._svd.transform(self._tfidf.transform(texts))
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / np.clip(norms, 1e-9, None)

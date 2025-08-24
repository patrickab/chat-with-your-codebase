"""Utilities for BM25 search over code chunks."""

import re
from typing import TYPE_CHECKING, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    import polars as pl


def _tokenize(text: str) -> list[str]:
    """Return lowercase word tokens from *text*."""
    return re.findall(r"\w+", text.lower())


def bm25_search(query: str, df: "pl.DataFrame", threshold_ratio: float = 0.5) -> list[dict[str, str]]:
    """Rank *df* code chunks by relevance to *query* using BM25.

    All results scoring below ``threshold_ratio`` times the best match are
    discarded. Empirically, BM25 scores drop off quickly; keeping only
    chunks within 50% of the top score filters out weak matches while
    still returning multiple strong candidates.
    """

    documents: list[list[str]] = []
    for row in df.iter_rows(named=True):
        # Emphasize function name and docstring for better relevance
        weighted_text = f"{row['name']} {row['name']} {row['docstring']} {row['docstring']} {row['code']}"
        documents.append(_tokenize(weighted_text))

    if not documents:
        return []

    bm25 = BM25Okapi(documents)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens) if query_tokens else np.zeros(len(documents))
    max_score = float(np.max(scores)) if len(scores) else 0.0
    if max_score <= 0:
        return []

    cutoff = max_score * threshold_ratio
    ranked_idx = np.argsort(scores)[::-1]
    results = []
    for i in ranked_idx:
        if scores[i] < cutoff:
            break
        results.append(df.row(int(i), named=True))
    return results


def build_context(chunks: Sequence[dict[str, str]]) -> str:
    """Build a context string from ranked *chunks*."""
    parts = []
    for c in chunks:
        parts.append(f"{c['full_name']}\n{c['code']}")
    return "\n\n".join(parts)

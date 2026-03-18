"""
rag_search.py
-------------
Minimal FAISS search for the livekit agent.

Only needs:
  - faiss-cpu
  - numpy
  - google-genai   (to embed the query string — FAISS is vector-only)

No scraper, no chunker, no langchain, no server.

uv add:
  uv add faiss-cpu numpy "google-genai>=1.0.0"

Files needed alongside this:
  rag_index/
    faiss.index      <- vector index
    metadata.pkl     <- chunk metadata (titles, urls, text, sections, etc.)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom unpickler — metadata.pkl was saved with DocumentChunk objects from
# chunker.py.  We don't want to ship chunker.py here, so we intercept the
# class lookup and return a plain SimpleNamespace instead.  Every attribute
# (doc_title, section, text, raw_content, …) is preserved exactly as pickled.
# ---------------------------------------------------------------------------

import types as _types  # stdlib types module — NOT google.genai.types


class _ChunkUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        # Intercept DocumentChunk (and anything else from chunker/scraper)
        if module in ("chunker", "scraper") or name == "DocumentChunk":
            return _types.SimpleNamespace
        return super().find_class(module, name)


@dataclass
class ChunkResult:
    rank:        int
    score:       float
    doc_title:   str
    section:     str
    doc_type:    str
    doc_url:     str
    doc_index:   int
    chunk_index: int
    text:        str       # formatted text (with section/title header)
    raw_content: str       # plain content without header


class RAGSearch:
    """
    Load a saved FAISS index and search it with natural-language queries.

    Parameters
    ----------
    index_dir       Path to the directory containing faiss.index + metadata.pkl
    gemini_api_key  Gemini API key — used ONLY to embed the query string
    model_name      Must match the model used when the index was built
                    Default: "gemini-embedding-001"
    dim             Embedding dimension — must match the index
                    Default: 3072
    """

    _INDEX_FILE = "faiss.index"
    _META_FILE  = "metadata.pkl"

    def __init__(
        self,
        index_dir:      str,
        gemini_api_key: str,
        model_name:     str = "gemini-embedding-001",
        dim:            int = 3072,
    ) -> None:
        self.model_name = model_name
        self.dim        = dim
        self._client    = genai.Client(api_key=gemini_api_key)

        path = Path(index_dir)
        if not path.exists():
            raise FileNotFoundError(f"Index directory not found: {index_dir!r}")

        logger.info("Loading FAISS index from %s …", index_dir)
        self._index = faiss.read_index(str(path / self._INDEX_FILE))

        with open(path / self._META_FILE, "rb") as f:
            state = _ChunkUnpickler(f).load()

        self._meta    = state["meta"]   # dict[int, DocumentChunk]
        self._next_id = state["next_id"]

        stats = self._stats()
        logger.info(
            "Index ready — %d chunks, %d documents",
            stats["total_chunks"], stats["total_documents"],
        )

    # -------------------------------------------------------------------------
    # Public search API
    # -------------------------------------------------------------------------

    def search(
        self,
        query:          str,
        top_k:          int           = 6,
        section_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
    ) -> list[ChunkResult]:
        """
        Embed `query` and return the top_k most similar chunks.

        Parameters
        ----------
        query           Natural-language question or phrase.
        top_k           How many results to return.
        section_filter  Only return chunks from this section, e.g. "research".
        doc_type_filter Only return chunks with this doc_type: "text",
                        "index", or "video_summary".
        """
        if not self._meta:
            logger.warning("Index is empty — returning []")
            return []

        # Embed query (one API call, ~200ms)
        q_vec = self._embed_query(query)

        # Fetch more than top_k so post-filtering still fills the quota
        fetch_k   = min(top_k * 5, len(self._meta))
        scores, ids = self._index.search(q_vec, fetch_k)

        results: list[ChunkResult] = []
        rank = 0
        for score, iid in zip(scores[0], ids[0]):
            if iid == -1:
                continue
            chunk = self._meta.get(int(iid))
            if chunk is None:
                continue
            if section_filter and getattr(chunk, "section", "") != section_filter:
                continue
            if doc_type_filter and getattr(chunk, "doc_type", "") != doc_type_filter:
                continue

            rank += 1
            results.append(ChunkResult(
                rank        = rank,
                score       = float(score),
                doc_title   = getattr(chunk, "doc_title",   ""),
                section     = getattr(chunk, "section",     ""),
                doc_type    = getattr(chunk, "doc_type",    ""),
                doc_url     = getattr(chunk, "doc_url",     ""),
                doc_index   = getattr(chunk, "doc_index",   0),
                chunk_index = getattr(chunk, "chunk_index", 0),
                text        = getattr(chunk, "text",        ""),
                raw_content = getattr(chunk, "raw_content", ""),
            ))
            if len(results) >= top_k:
                break

        if results:
            logger.info(
                "[search] %d result(s)  top=%.4f  query=%r",
                len(results), results[0].score, query[:80],
            )
        else:
            logger.info("[search] no results for query=%r", query[:80])

        return results

    def stats(self) -> dict:
        return self._stats()

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _embed_query(self, query: str) -> np.ndarray:
        """Single Gemini embedding call for a query string."""
        result = self._client.models.embed_content(
            model=self.model_name,
            contents=[query],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.dim,
            ),
        )
        vec = np.array([result.embeddings[0].values], dtype="float32")
        # L2-normalise so inner-product == cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _stats(self) -> dict:
        sections: dict[str, int] = {}
        titles:   set[str]       = set()
        for chunk in self._meta.values():
            sec = getattr(chunk, "section", "?")
            sections[sec] = sections.get(sec, 0) + 1
            titles.add(getattr(chunk, "doc_title", "?"))
        return {
            "total_chunks":    len(self._meta),
            "total_documents": len(titles),
            "sections":        sections,
        }
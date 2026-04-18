"""Unified RAG — NVIDIA embeddings + FAISS over refugee camp records plus
text documents (NDMA SOPs, landslide atlas, relief camp protocols, ...).

Loads the corpus once per process; queries re-use the index across reruns.
"""
from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # faiss is optional at import time; the fallback path still works.
    import faiss  # type: ignore
except ImportError:  # pragma: no cover — faiss required at runtime
    faiss = None  # type: ignore

from openai import OpenAI

from app.config import cfg, nvidia_embedding_api_key, resolve_path

logger = logging.getLogger(__name__)

_FALLBACK_DIM = 1024


@lru_cache(maxsize=1)
def _get_corpus() -> List[Dict[str, Any]]:
    """Load refugee camps + text docs into a unified list of RAG documents."""
    docs: List[Dict[str, Any]] = []

    # Refugee camp records
    camps_path = resolve_path(str(cfg("data.refugee_camps_file", "data/refugee_camps.json")))
    for camp in _load_camps(camps_path):
        text = (
            f"{camp['name']} is a {camp['type']} located at "
            f"{camp['lat']:.4f}, {camp['lon']:.4f}. {camp.get('description', '')}"
        ).strip()
        docs.append(
            {
                "id": f"camp:{camp['name']}",
                "source": "refugee_camps",
                "title": camp["name"],
                "text": text,
                "metadata": {
                    "type": camp["type"],
                    "lat": camp["lat"],
                    "lon": camp["lon"],
                    "kind": "camp",
                },
            }
        )

    # Text documents (docs_dir)
    docs_dir = resolve_path(str(cfg("data.docs_dir", "data/docs")))
    for text_doc in _load_text_docs(docs_dir):
        docs.append(text_doc)

    max_docs = int(cfg("retrieval.max_docs", 200))
    return docs[:max_docs]


def _load_camps(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        logger.warning("refugee_camps file missing: %s", path)
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("refugee_camps file unreadable: %s", path)
        return []

    records: List[Any]
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("camps") or payload.get("data") or []
    else:
        records = []

    camps: List[Dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        lat = _safe_float(item.get("lat"))
        lon = _safe_float(item.get("lon"))
        if lat is None or lon is None:
            continue
        camps.append(
            {
                "name": str(item.get("name", "Unknown Camp")),
                "lat": lat,
                "lon": lon,
                "type": str(item.get("type", "refugee_camp")),
                "description": str(item.get("description", "")),
            }
        )
    return camps


def _load_text_docs(docs_dir: Path) -> List[Dict[str, Any]]:
    if not docs_dir.exists():
        return []

    documents: List[Dict[str, Any]] = []
    for txt_path in sorted(docs_dir.glob("*.txt")):
        try:
            text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        if not text:
            continue

        meta = _load_meta_sidecar(txt_path)
        documents.append(
            {
                "id": f"doc:{txt_path.stem}",
                "source": meta.get("source", "local_docs"),
                "title": meta.get("title", txt_path.stem.replace("_", " ").title()),
                "text": text,
                "metadata": {
                    "kind": "document",
                    "url": meta.get("url"),
                    "published_at": meta.get("published_at"),
                    "file": txt_path.name,
                },
            }
        )
    return documents


def _load_meta_sidecar(txt_path: Path) -> Dict[str, Any]:
    meta_path = txt_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@lru_cache(maxsize=1)
def _get_index() -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    corpus = _get_corpus()
    if not corpus:
        return None, []
    if faiss is None:
        logger.warning("faiss not importable — RAG will fall back to linear scoring")
        return None, corpus

    texts = [doc["text"][:2000] for doc in corpus]
    matrix = np.asarray([get_embedding(text) for text in texts], dtype=np.float32)
    if matrix.size == 0:
        return None, corpus

    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return index, corpus


def search(query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return ranked RAG documents for a natural-language query."""
    k = int(top_k or cfg("retrieval.top_k", 5))
    normalized = (query or "").strip()
    if not normalized:
        logger.warning("Empty query received")
        return []

    max_chars = int(cfg("retrieval.max_query_chars", 2000))
    normalized = normalized[:max_chars]
    
    # Debug logging
    logger.debug(f"Searching for query: {normalized[:50]}...")

    index, corpus = _get_index()
    if not corpus:
        logger.warning("No corpus loaded for search")
        return []

    query_vec = np.asarray([get_embedding(normalized)], dtype=np.float32)
    if query_vec.size == 0:
        logger.warning("Query embedding is empty")
        return []

    if index is not None:
        distances, indices = index.search(query_vec, min(k, len(corpus)))
        results: List[Dict[str, Any]] = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(corpus):
                continue
            doc = corpus[int(idx)]
            results.append(_format_result(doc, float(distance)))
        
        # Debug: log top result
        if results:
            logger.debug(f"Top result for '{normalized[:30]}...': {results[0]['title']} (score: {results[0]['score']})")
        return results

    # Fallback: cosine-like ranking via numpy when faiss is unavailable.
    logger.debug("Using numpy fallback for search (FAISS unavailable)")
    doc_matrix = np.asarray([get_embedding(doc["text"][:2000]) for doc in corpus], dtype=np.float32)
    diffs = doc_matrix - query_vec
    scores = np.linalg.norm(diffs, axis=1)
    ranked_idx = np.argsort(scores)[:k]
    return [_format_result(corpus[int(i)], float(scores[int(i)])) for i in ranked_idx]


def retrieve_camps_only(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Return only camp-kind RAG results as camp dicts (for orchestration context)."""
    camps: List[Dict[str, Any]] = []
    for result in search(query, top_k=top_k * 3):
        metadata = result.get("metadata", {}) or {}
        if metadata.get("kind") != "camp":
            continue
        camps.append(
            {
                "name": result.get("title", "Unknown Camp"),
                "type": metadata.get("type", "refugee_camp"),
                "lat": metadata.get("lat"),
                "lon": metadata.get("lon"),
                "text": result.get("text", ""),
            }
        )
        if len(camps) >= top_k:
            break
    return camps


def _format_result(doc: Dict[str, Any], distance: float) -> Dict[str, Any]:
    return {
        "id": doc["id"],
        "source": doc.get("source", "unknown"),
        "title": doc.get("title", ""),
        "text": doc.get("text", ""),
        "metadata": doc.get("metadata", {}),
        "score": round(distance, 4),
    }


# --- NVIDIA embedding helpers -------------------------------------------------


@lru_cache(maxsize=512)
def _embed_cached(text: str) -> Tuple[float, ...]:
    if not text:
        return _fallback_vector(text)

    api_key = nvidia_embedding_api_key()
    if not api_key:
        logger.info("NVIDIA_API_KEY missing — using deterministic fallback embeddings.")
        return _fallback_vector(text)

    base_url = str(cfg("embeddings.base_url", "https://integrate.api.nvidia.com/v1"))
    model = str(cfg("embeddings.model", "nvidia/nv-embedqa-e5-v5"))

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=25.0)
        response = client.embeddings.create(
            input=[text],
            model=model,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"},
        )
        if response.data and response.data[0].embedding:
            return tuple(float(v) for v in response.data[0].embedding)
        logger.warning("NVIDIA embedding response empty; using fallback.")
    except Exception as exc:  # pragma: no cover — network-dependent
        logger.warning("NVIDIA embedding request failed: %s — falling back", exc)

    return _fallback_vector(text)


def get_embedding(text: str) -> List[float]:
    return list(_embed_cached(text.strip() if text else ""))


def clear_embedding_cache() -> None:
    """Clear embedding + corpus + index caches. Call after reloading .env."""
    _embed_cached.cache_clear()
    _fallback_vector.cache_clear()
    _get_corpus.cache_clear()
    _get_index.cache_clear()


@lru_cache(maxsize=512)
def _fallback_vector(text: str) -> Tuple[float, ...]:
    dim = int(cfg("embeddings.fallback_dim", _FALLBACK_DIM))
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    vector = rng.random(dim, dtype=np.float32)
    norm = math.sqrt(float((vector ** 2).sum())) or 1.0
    return tuple(float(v / norm) for v in vector)


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

"""Top-level orchestration — ties together infra, hazards, RAG, scoring,
and NVIDIA reasoning into a single `run_assistant()` call for the Streamlit app.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app import hazards, nearby_infra, rag, reasoning, risk_scoring

logger = logging.getLogger(__name__)


def run_assistant(
    lat: float,
    lon: float,
    query: str,
    district: Optional[str] = None,
    state: Optional[str] = None,
    radius_m: int = 5000,
    top_k_docs: int = 5,
) -> Dict[str, Any]:
    """
    Orchestrate the full pipeline:
    1. OSM infrastructure profile (rich multi-category)
    2. Hazard ingestion (weather + alerts + history)
    3. RAG retrieval (camps + docs via NVIDIA embeddings)
    4. Deterministic district risk score
    5. NVIDIA Nemotron reasoning (with rule-based fallback)

    Always returns a populated dict — errors land in `warnings`.
    """
    warnings: List[str] = []

    # 1. Infrastructure
    try:
        infra_profile = nearby_infra.get_infra_profile(lat, lon, radius_m=radius_m)
    except Exception as exc:  # pragma: no cover — network-dependent
        logger.warning("infra profile failed: %s", exc)
        warnings.append(f"Infrastructure fetch failed: {exc}")
        infra_profile = nearby_infra.get_infra_profile(lat, lon, radius_m=radius_m)  # returns empty on error

    if infra_profile.get("error"):
        warnings.append(f"Infrastructure: {infra_profile['error']}")

    # 2. Hazard context
    try:
        hazard_context = hazards.get_hazard_context(lat, lon, query)
    except Exception as exc:
        logger.warning("hazard context failed: %s", exc)
        warnings.append(f"Hazard ingestion failed: {exc}")
        hazard_context = {"weather": {}, "hazard": {}, "alerts": [], "history": []}

    # 3. RAG retrieval
    try:
        retrieved_docs = rag.search(query, top_k=top_k_docs)
    except Exception as exc:
        logger.warning("rag search failed: %s", exc)
        warnings.append(f"RAG search failed: {exc}")
        retrieved_docs = []

    nearby_camps = [
        doc for doc in retrieved_docs
        if isinstance(doc.get("metadata"), dict) and doc["metadata"].get("kind") == "camp"
    ][:3]

    # 4. Deterministic district score
    try:
        scoring = risk_scoring.compute_district_risk(district, state, hazard_context)
    except Exception as exc:
        logger.warning("risk scoring failed: %s", exc)
        warnings.append(f"Risk scoring failed: {exc}")
        scoring = {"district": district, "state": state, "score": 0.0, "label": "Unknown", "components": {}}

    # 5. LLM reasoning
    rag_context = {
        "structured_data": hazard_context,
        "retrieved_docs": retrieved_docs,
        "infra": infra_profile,
        "risk_scoring": scoring,
    }
    try:
        risk_analysis = reasoning.generate_risk_response(query, rag_context)
    except Exception as exc:
        logger.warning("risk reasoning failed: %s", exc)
        warnings.append(f"Risk reasoning failed: {exc}")
        risk_analysis = reasoning.generate_risk_response(query, rag_context)  # will fall back internally

    return {
        "location": {"lat": lat, "lon": lon, "district": district, "state": state},
        "query": query,
        "infra": infra_profile,
        "hazards": hazard_context,
        "retrieved_docs": retrieved_docs,
        "nearby_camps": nearby_camps,
        "risk_scoring": scoring,
        "risk_analysis": risk_analysis,
        "warnings": warnings,
    }


def answer_question(query: str, top_k_docs: int = 5) -> Dict[str, Any]:
    """Chat-style RAG Q&A — independent of location.

    Returns the full reasoning payload plus the raw retrieved docs for display.
    """
    try:
        retrieved_docs = rag.search(query, top_k=top_k_docs)
    except Exception as exc:
        logger.warning("rag search failed: %s", exc)
        retrieved_docs = []

    reply = reasoning.generate_chat_answer(query, retrieved_docs)
    reply["retrieved_docs"] = retrieved_docs
    return reply

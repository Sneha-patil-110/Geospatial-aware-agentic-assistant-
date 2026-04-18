"""NVIDIA Nemotron reasoning layer with rule-based fallback.

Produces grounded risk analysis AND grounded chat answers for the combined
Streamlit app. Always returns something — even when NVIDIA keys are missing
or the LLM output is unparseable.

Key design choices:
  * enable_thinking is disabled by default — Nemotron's chain-of-thought
    eats most of the token budget, producing empty `content` at low max_tokens.
  * Parsing is lenient: strips ```json fences, handles content that lives in
    reasoning_content instead of content, repairs unterminated JSON.
  * On parse failure we do a strict retry with a deterministic prompt before
    giving up to the rule-based fallback.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.config import cfg, nvidia_reasoning_api_key

logger = logging.getLogger(__name__)


RISK_SYSTEM_PROMPT = (
    "You are a civilian safety and disaster risk assistant. "
    "Use ONLY the structured context provided in the user message. "
    "Do not invent facts. If data is missing, say so. "
    "Return ONLY a JSON object, no prose, no markdown fences. "
    "Schema: {\"risk_level\": \"Low\"|\"Medium\"|\"High\", "
    "\"explanation\": \"...\", \"recommended_action\": \"...\"}"
)

CHAT_SYSTEM_PROMPT = (
    "You are a Civilian Safety Zone assistant. "
    "Answer the user's question concisely using ONLY the retrieved documents "
    "and structured data provided. Every factual claim must reference a source "
    "title in brackets, e.g. [NDMA Flood Guidelines]. "
    "If no documents answer the question, respond: "
    '"I do not have enough grounded data to answer that."'
)

DEFAULT_MAX_TOKENS = 1024


# ─── Public API ───────────────────────────────────────────────────────────────


def generate_risk_response(query: str, rag_context: Dict[str, Any]) -> Dict[str, Any]:
    """Structured risk analysis dict — always returned, never None."""
    context = rag_context if isinstance(rag_context, dict) else {}
    prompt = _build_risk_prompt(query or "Provide a hazard risk summary.", context)

    api_key = nvidia_reasoning_api_key()
    if not api_key:
        logger.warning("NVIDIA reasoning API key missing")
        return _risk_fallback("NVIDIA_API_KEY missing", context)

    client = _client(api_key)
    model = str(cfg("reasoning.model", "nvidia/nemotron-3-nano-30b-a3b"))
    temperature = float(cfg("reasoning.temperature", 0.2))
    max_tokens = int(cfg("reasoning.max_tokens", DEFAULT_MAX_TOKENS))

    # First attempt — thinking off, moderate max_tokens.
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": RISK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = _extract_content(response)
        logger.info("risk LLM pass-1 content[:300]=%r", content[:300])
        parsed = _parse_risk_json(content)
        if parsed:
            parsed["used_llm"] = True
            parsed["raw"] = content[:600]
            return parsed
    except Exception as exc:
        logger.warning("NVIDIA risk pass-1 failed: %s", exc, exc_info=True)

    # Strict retry — zero temperature, explicit JSON-only instruction.
    try:
        retry_content = _retry_risk_json_only(client, prompt, model, max_tokens)
        logger.info("risk LLM pass-2 content[:300]=%r", retry_content[:300])
        parsed = _parse_risk_json(retry_content)
        if parsed:
            parsed["used_llm"] = True
            parsed["raw"] = retry_content[:600]
            return parsed
    except Exception as exc:
        logger.warning("NVIDIA risk pass-2 failed: %s", exc, exc_info=True)

    return _risk_fallback("LLM output was empty or unparseable", context)


def generate_chat_answer(query: str, rag_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Chat-style grounded answer built from RAG documents."""
    clean_query = (query or "").strip()
    if not clean_query:
        return {"answer": "Please ask a question.", "citations": [], "used_llm": False}

    if not rag_docs:
        return {
            "answer": "I do not have enough grounded data to answer that.",
            "citations": [],
            "used_llm": False,
        }

    api_key = nvidia_reasoning_api_key()
    citations = _collect_citations(rag_docs)
    if not api_key:
        logger.warning("NVIDIA reasoning API key missing for chat")
        return {
            "answer": _rule_based_chat(clean_query, rag_docs),
            "citations": citations,
            "used_llm": False,
        }

    prompt = _build_chat_prompt(clean_query, rag_docs)
    model = str(cfg("reasoning.model", "nvidia/nemotron-3-nano-30b-a3b"))
    max_tokens = int(cfg("reasoning.max_tokens", DEFAULT_MAX_TOKENS))

    try:
        client = _client(api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=float(cfg("reasoning.temperature", 0.2)),
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = _extract_content(response).strip()
        logger.info("chat LLM content length=%d", len(content))
        if content:
            return {"answer": content, "citations": citations, "used_llm": True}
        logger.warning("Chat model returned empty content")
    except Exception as exc:
        logger.warning("NVIDIA chat call failed: %s", exc, exc_info=True)

    return {
        "answer": _rule_based_chat(clean_query, rag_docs),
        "citations": citations,
        "used_llm": False,
    }


# ─── Prompt builders ───────────────────────────────────────────────────────────


def _build_risk_prompt(query: str, context: Dict[str, Any]) -> str:
    structured = context.get("structured_data") if isinstance(context.get("structured_data"), dict) else {}
    compact = {
        "weather": (structured.get("weather") or {}),
        "hazard": (structured.get("hazard") or {}),
        "alerts": _truncate_list(structured.get("alerts"), 5),
        "history": _truncate_list(structured.get("history"), 5),
    }
    infra = context.get("infra") or {}
    if isinstance(infra, dict):
        compact_infra = {
            "category_counts": infra.get("category_counts", {}),
            "safety_score": infra.get("safety_score", {}),
            "emergency_shortlist": infra.get("emergency_shortlist", {}),
        }
    else:
        compact_infra = {"items": _truncate_list(infra, 6)}

    compact_docs = [
        {
            "title": doc.get("title", ""),
            "source": doc.get("source", ""),
            "snippet": (doc.get("text") or "")[:280],
        }
        for doc in _truncate_list(context.get("retrieved_docs"), 5)
    ]

    risk_scoring = context.get("risk_scoring") or {}

    return (
        f"User query:\n{query}\n\n"
        f"Structured hazard data:\n{json.dumps(compact, ensure_ascii=True)}\n\n"
        f"Infrastructure summary:\n{json.dumps(compact_infra, ensure_ascii=True)}\n\n"
        f"Deterministic district risk score:\n{json.dumps(risk_scoring, ensure_ascii=True)}\n\n"
        f"Retrieved documents:\n{json.dumps(compact_docs, ensure_ascii=True)}\n\n"
        "Return ONLY this exact JSON shape (no markdown, no prose):\n"
        '{"risk_level":"Low|Medium|High","explanation":"...","recommended_action":"..."}'
    )


def _build_chat_prompt(query: str, docs: List[Dict[str, Any]]) -> str:
    packed = []
    for idx, doc in enumerate(docs[:6], start=1):
        packed.append(
            f"[{idx}] Title: {doc.get('title', 'Untitled')}\n"
            f"    Source: {doc.get('source', 'unknown')}\n"
            f"    Snippet: {(doc.get('text') or '')[:600]}"
        )
    return (
        f"Question:\n{query}\n\n"
        "Retrieved documents:\n" + "\n\n".join(packed) + "\n\n"
        "Respond in 2-4 short paragraphs. Cite using the document titles in brackets."
    )


# ─── Fallbacks ────────────────────────────────────────────────────────────────


def _risk_fallback(reason: str, context: Dict[str, Any]) -> Dict[str, Any]:
    structured = context.get("structured_data") if isinstance(context, dict) else {}
    hazard = (structured or {}).get("hazard") or {}
    hazard_type = str(hazard.get("hazard_type", "")).lower()
    severity = str(hazard.get("severity", "")).lower()

    alerts = _truncate_list((structured or {}).get("alerts"), 5)
    infra = context.get("infra") if isinstance(context, dict) else {}
    infra_profile = infra if isinstance(infra, dict) else {}
    category_counts = infra_profile.get("category_counts", {}) or {}
    total_items = sum(int(v) for v in category_counts.values() if isinstance(v, (int, float)))

    hazards_present = (
        hazard_type not in ("", "none", "unknown")
        or severity in ("medium", "high")
        or bool(alerts)
    )
    low_infrastructure = total_items < 3

    if hazards_present:
        return {
            "risk_level": "High",
            "explanation": (
                f"Fallback reasoning used ({reason}). Active hazard signals or alerts are "
                "present, so the area is treated as high risk."
            ),
            "recommended_action": (
                "Issue caution advisory, reroute vulnerable residents toward nearest "
                "shelters, and monitor official alerts every 30 minutes."
            ),
            "used_llm": False,
        }

    if low_infrastructure:
        return {
            "risk_level": "Medium",
            "explanation": (
                f"Fallback reasoning used ({reason}). Hazard signals are limited but "
                "nearby emergency infrastructure is sparse, so risk cannot be dismissed."
            ),
            "recommended_action": (
                "Prepare contingency movement plan, pre-stock water and food, and track "
                "official bulletins."
            ),
            "used_llm": False,
        }

    return {
        "risk_level": "Low",
        "explanation": (
            f"Fallback reasoning used ({reason}). No strong hazard signals detected and "
            "reasonable emergency infrastructure is nearby."
        ),
        "recommended_action": "Continue routine monitoring and follow official advisories.",
        "used_llm": False,
    }


def _rule_based_chat(query: str, docs: List[Dict[str, Any]]) -> str:
    header = "Grounded answer (rule-based, no LLM key):\n\n"
    bullets: List[str] = []
    for doc in docs[:5]:
        title = doc.get("title") or "Untitled"
        snippet = (doc.get("text") or "").replace("\n", " ")[:220]
        bullets.append(f"- [{title}] {snippet}")
    if not bullets:
        return "I do not have enough grounded data to answer that."
    return header + "\n".join(bullets)


# ─── helpers ──────────────────────────────────────────────────────────────────


def _client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=str(cfg("reasoning.base_url", "https://integrate.api.nvidia.com/v1")),
        timeout=60.0,
    )


def _retry_risk_json_only(client: OpenAI, prompt: str, model: str, max_tokens: int) -> str:
    strict_system = (
        "Return ONLY a JSON object. No analysis, no markdown fences, no extra text. "
        "Use keys exactly: risk_level, explanation, recommended_action."
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": strict_system},
            {"role": "user", "content": prompt},
        ],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return _extract_content(response)


def _extract_content(response: Any) -> str:
    """Pull text out of an OpenAI-style response, preferring `content` but
    falling back to `reasoning_content` when thinking mode accidentally stays on.
    """
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return ""

    content = getattr(message, "content", None) or ""
    if content and content.strip():
        return content.strip()

    # Nemotron sometimes puts the whole response into reasoning_content.
    reasoning = getattr(message, "reasoning_content", None) or ""
    if reasoning and reasoning.strip():
        return reasoning.strip()

    # Dict-style payload (older clients).
    if isinstance(message, dict):
        return (message.get("content") or message.get("reasoning_content") or "").strip()

    return ""


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _parse_risk_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    # 1. Strip ```json ... ``` fences if present.
    cleaned = _FENCE_RE.sub("", text).strip()

    # 2. Try direct JSON load.
    payload = _try_load(cleaned)
    # 3. Fall back to the first balanced {...} block.
    if payload is None:
        payload = _try_load(_first_json_object(cleaned))

    if not isinstance(payload, dict):
        return None

    risk_raw = (
        payload.get("risk_level")
        or payload.get("riskLevel")
        or payload.get("risk")
        or ""
    )
    risk = str(risk_raw).strip().capitalize()
    if risk in ("Low risk", "Medium risk", "High risk"):
        risk = risk.split()[0]

    explanation = str(
        payload.get("explanation")
        or payload.get("reason")
        or payload.get("summary")
        or ""
    ).strip()
    action = str(
        payload.get("recommended_action")
        or payload.get("action")
        or payload.get("recommendation")
        or ""
    ).strip()

    if risk not in ("Low", "Medium", "High"):
        return None
    if not explanation:
        explanation = "No explanation provided by model."
    if not action:
        action = "Monitor official advisories and local infrastructure status."

    return {
        "risk_level": risk,
        "explanation": explanation,
        "recommended_action": action,
    }


def _try_load(text: str) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _first_json_object(text: str) -> str:
    """Extract the first balanced {...} block from a string.
    Handles braces inside quoted strings correctly.
    """
    if not text:
        return ""
    depth = 0
    in_str = False
    escape = False
    start = -1
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return text[start:i + 1]
    return ""


def _collect_citations(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    seen = set()
    for doc in docs:
        title = (doc.get("title") or "").strip()
        if not title or title in seen:
            continue
        seen.add(title)
        citations.append(
            {
                "title": title,
                "source": doc.get("source", "unknown"),
                "id": doc.get("id", ""),
            }
        )
    return citations


def _truncate_list(value: Any, cap: int) -> List[Any]:
    if not isinstance(value, list):
        return []
    return value[:cap]

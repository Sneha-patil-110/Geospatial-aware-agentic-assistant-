"""Deterministic district-level risk scoring using flood + landslide priors.

Ported from safety-zone's insights/scoring.py, but event evidence now comes
from hazard_ingestion (weather + alerts + history) rather than a SQLite DB.
"""
from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from typing import Any, Dict, List, Optional

from app.config import cfg, resolve_path

logger = logging.getLogger(__name__)

SEVERITY_WEIGHT: Dict[str, float] = {"low": 0.1, "medium": 0.3, "moderate": 0.3, "high": 0.6, "critical": 1.0}


def compute_district_risk(
    district: Optional[str],
    state: Optional[str],
    hazard_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Produce a score in [0, 1] plus a Safe/Moderate/Unsafe label.

    hazard_context is the structured payload from hazards.get_hazard_context.
    When a district is empty or missing from the priors, defaults are used.
    """
    district = (district or "").strip()
    state = (state or "").strip() or None

    flood_prior = _lookup_prior(district, "flood_inventory.geojson", default=0.1)
    landslide_prior = _lookup_prior(district, "landslide_atlas.geojson", default=0.05)

    event_score_norm = _event_score_from_hazard(hazard_context)

    flood_w = float(cfg("scoring.flood_prior_weight", 0.3))
    landslide_w = float(cfg("scoring.landslide_prior_weight", 0.3))
    event_w = max(0.0, 1.0 - flood_w - landslide_w)

    final = (
        event_w * event_score_norm
        + flood_w * flood_prior
        + landslide_w * landslide_prior
    )
    final = min(max(final, 0.0), 1.0)

    safe_threshold = float(cfg("scoring.safe_threshold", 0.3))
    unsafe_threshold = float(cfg("scoring.unsafe_threshold", 0.7))
    if final < safe_threshold:
        label = "Safe"
    elif final < unsafe_threshold:
        label = "Moderate"
    else:
        label = "Unsafe"

    return {
        "district": district or None,
        "state": state,
        "score": round(final, 4),
        "label": label,
        "components": {
            "event_score": round(event_score_norm, 4),
            "flood_prior": round(flood_prior, 4),
            "landslide_prior": round(landslide_prior, 4),
            "event_weight": round(event_w, 4),
            "flood_weight": round(flood_w, 4),
            "landslide_weight": round(landslide_w, 4),
        },
    }


def _event_score_from_hazard(hazard_context: Optional[Dict[str, Any]]) -> float:
    if not isinstance(hazard_context, dict):
        return 0.0

    score = 0.0

    hazard = hazard_context.get("hazard") or {}
    if isinstance(hazard, dict):
        severity = str(hazard.get("severity", "")).strip().lower()
        if severity in SEVERITY_WEIGHT:
            score += SEVERITY_WEIGHT[severity]

    for alert in hazard_context.get("alerts") or []:
        if not isinstance(alert, dict):
            continue
        severity = str(alert.get("severity", "")).strip().lower()
        score += SEVERITY_WEIGHT.get(severity, 0.1) * 0.5  # alerts weighted half

    for record in hazard_context.get("history") or []:
        if not isinstance(record, dict):
            continue
        severity = str(record.get("severity", "")).strip().lower()
        score += SEVERITY_WEIGHT.get(severity, 0.1) * 0.25  # history weighted lower

    # Saturate to [0, 1) so priors can still meaningfully contribute.
    return 1.0 - math.exp(-score)


@lru_cache(maxsize=4)
def _load_priors(filename: str) -> Dict[str, float]:
    geo_dir = resolve_path(str(cfg("data.geo_dir", "data/geo")))
    path = geo_dir / filename
    if not path.exists():
        logger.info("Geo prior file missing: %s", path)
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Geo prior file unreadable: %s", path)
        return {}

    priors: Dict[str, float] = {}
    for feature in payload.get("features", []) if isinstance(payload, dict) else []:
        props = feature.get("properties", {}) if isinstance(feature, dict) else {}
        name = str(props.get("district") or props.get("name") or "").strip().lower()
        if not name:
            continue
        raw = props.get("risk_score")
        if raw is None:
            raw = props.get("frequency", 0.1)
        try:
            priors[name] = min(max(float(raw), 0.0), 1.0)
        except (TypeError, ValueError):
            priors[name] = 0.1
    return priors


def _lookup_prior(district: str, filename: str, default: float) -> float:
    if not district:
        return default
    priors = _load_priors(filename)
    return priors.get(district.lower(), default)

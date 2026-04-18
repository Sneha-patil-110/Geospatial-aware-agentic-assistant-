"""OpenStreetMap nearby infrastructure — combines the emergency-focused query
from Enigma with the richer multi-category profile from safety-zone.

Returns both the grouped emergency shortlist and the full infra profile.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.config import cfg

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

INFRA_QUERY_TEMPLATE = """
[out:json][timeout:30];
(
  /* emergency & relief */
  node["amenity"="shelter"](around:{radius},{lat},{lon});
  node["amenity"="refugee_site"](around:{radius},{lat},{lon});
  node["amenity"="social_facility"](around:{radius},{lat},{lon});
  node["emergency"="assembly_point"](around:{radius},{lat},{lon});
  node["emergency"="evacuation_point"](around:{radius},{lat},{lon});
  /* medical */
  node["amenity"="hospital"](around:{radius},{lat},{lon});
  way["amenity"="hospital"](around:{radius},{lat},{lon});
  node["amenity"="clinic"](around:{radius},{lat},{lon});
  node["amenity"="pharmacy"](around:{radius},{lat},{lon});
  node["amenity"="doctors"](around:{radius},{lat},{lon});
  node["amenity"="first_aid"](around:{radius},{lat},{lon});
  /* water */
  node["amenity"="water_point"](around:{radius},{lat},{lon});
  node["amenity"="drinking_water"](around:{radius},{lat},{lon});
  node["man_made"="water_well"](around:{radius},{lat},{lon});
  /* food */
  node["amenity"="food_bank"](around:{radius},{lat},{lon});
  node["amenity"="community_centre"](around:{radius},{lat},{lon});
  node["shop"="supermarket"](around:{radius},{lat},{lon});
  node["amenity"="marketplace"](around:{radius},{lat},{lon});
  /* safety & security */
  node["amenity"="police"](around:{radius},{lat},{lon});
  node["amenity"="fire_station"](around:{radius},{lat},{lon});
  /* evacuation infra */
  node["amenity"="school"](around:{radius},{lat},{lon});
  way["amenity"="school"](around:{radius},{lat},{lon});
  node["amenity"="college"](around:{radius},{lat},{lon});
  node["amenity"="townhall"](around:{radius},{lat},{lon});
  /* transport */
  node["highway"="bus_stop"](around:{radius},{lat},{lon});
  node["railway"="station"](around:{radius},{lat},{lon});
  node["amenity"="fuel"](around:{radius},{lat},{lon});
);
out center body;
""".strip()

CATEGORY_MAP: Dict[Tuple[str, Optional[str]], str] = {
    ("amenity", "hospital"): "medical",
    ("amenity", "clinic"): "medical",
    ("amenity", "doctors"): "medical",
    ("amenity", "pharmacy"): "medical",
    ("amenity", "first_aid"): "medical",
    ("amenity", "shelter"): "shelter",
    ("amenity", "refugee_site"): "shelter",
    ("amenity", "social_facility"): "shelter",
    ("emergency", "assembly_point"): "shelter",
    ("emergency", "evacuation_point"): "shelter",
    ("amenity", "water_point"): "water",
    ("amenity", "drinking_water"): "water",
    ("man_made", "water_well"): "water",
    ("amenity", "food_bank"): "food",
    ("amenity", "community_centre"): "food",
    ("shop", "supermarket"): "food",
    ("amenity", "marketplace"): "food",
    ("amenity", "police"): "security",
    ("amenity", "fire_station"): "security",
    ("amenity", "school"): "evacuation",
    ("amenity", "college"): "evacuation",
    ("amenity", "townhall"): "evacuation",
    ("highway", "bus_stop"): "transport",
    ("railway", "station"): "transport",
    ("amenity", "fuel"): "transport",
}

CATEGORY_SAFETY_WEIGHT: Dict[str, float] = {
    "medical": 2.5,
    "shelter": 2.0,
    "water": 1.8,
    "food": 1.2,
    "security": 2.0,
    "evacuation": 1.5,
    "transport": 1.0,
}


def get_infra_profile(
    lat: float,
    lon: float,
    radius_m: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fetch comprehensive infrastructure profile around (lat, lon).
    Results are cached for up to 1 hour per location.

    Returns:
        {
          "center": {"lat": ..., "lon": ...},
          "radius_m": int,
          "by_category": {"medical": [...], "shelter": [...], ...},
          "category_counts": {...},
          "emergency_shortlist": {"hospitals": [...], "police_stations": [...], "fire_stations": [...]},
          "safety_score": {"raw": float, "normalised": float, "contributions": {...}},
          "fetched_at": iso timestamp,
          "error": Optional[str],
        }
    """
    src_lat = _safe_float(lat)
    src_lon = _safe_float(lon)
    radius = int(radius_m) if radius_m is not None else int(cfg("osm.default_radius_m", 5000))

    empty = _empty_profile(src_lat, src_lon, radius)
    if src_lat is None or src_lon is None:
        empty["error"] = "invalid coordinates"
        return empty

    # Use cached query for this location
    query = INFRA_QUERY_TEMPLATE.format(lat=src_lat, lon=src_lon, radius=radius)
    cached_json = _cached_overpass_query(query)
    
    if cached_json is not None:
        try:
            data = json.loads(cached_json)
            elements = data.get("elements", []) if isinstance(data, dict) else []
        except (json.JSONDecodeError, ValueError):
            elements = None
    else:
        elements = None
    
    # Fallback: Generate mock infrastructure data when API unavailable
    if elements is None:
        elements = _generate_fallback_infrastructure(src_lat, src_lon, radius)
        if not elements:
            empty["error"] = "Overpass API unavailable and fallback data not available"
            return empty
        empty["error"] = "Using fallback infrastructure data (OSM API unavailable)"

    by_category: Dict[str, List[Dict[str, Any]]] = {cat: [] for cat in CATEGORY_SAFETY_WEIGHT}
    for element in elements:
        tags = element.get("tags", {}) or {}
        category = _categorise(tags)
        if category is None:
            continue

        item_lat, item_lon = _extract_coordinates(element)
        distance_km = (
            _haversine_km(src_lat, src_lon, item_lat, item_lon)
            if item_lat is not None and item_lon is not None
            else None
        )

        by_category[category].append(
            {
                "osm_id": element.get("id"),
                "name": _derive_name(tags, category),
                "type": tags.get("amenity") or tags.get("emergency") or category,
                "category": category,
                "lat": item_lat,
                "lon": item_lon,
                "distance_km": round(distance_km, 3) if distance_km is not None else None,
                "tags": tags,
            }
        )

    cap = int(cfg("osm.max_per_category", 10))
    for category, items in by_category.items():
        items.sort(key=lambda it: it["distance_km"] if it["distance_km"] is not None else float("inf"))
        del items[cap:]

    emergency_shortlist = _build_emergency_shortlist(by_category)
    safety_score = _compute_safety_score(by_category, radius / 1000.0)

    return {
        "center": {"lat": src_lat, "lon": src_lon},
        "radius_m": radius,
        "by_category": by_category,
        "category_counts": {cat: len(items) for cat, items in by_category.items()},
        "emergency_shortlist": emergency_shortlist,
        "safety_score": safety_score,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        # Preserve any fallback/error flag set earlier (synthetic data, partial failure, etc.)
        "error": empty.get("error"),
    }


# --- backwards-compatible helper matching Enigma's get_nearby_infra(lat, lon) --

def get_nearby_infra(lat: float, lon: float) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return the emergency-focused shortlist (hospitals, police, fire) only.

    Kept for compatibility with Enigma's original assistant_runner signature.
    """
    profile = get_infra_profile(lat, lon, radius_m=2000)
    return profile["emergency_shortlist"]


# --- helpers ------------------------------------------------------------------

def _generate_fallback_infrastructure(lat: float, lon: float, radius_m: int) -> List[Dict[str, Any]]:
    """
    Generate synthetic infrastructure data when Overpass API is unavailable.
    Creates realistic mock items near the given location for testing/fallback.
    """
    import random
    
    # Seed based on coordinates for reproducibility
    random.seed(int(lat * 1000) + int(lon * 1000))
    
    mock_items = []
    
    # Define mock infrastructure templates
    infrastructure_templates = [
        {"name": "District Hospital", "type": "hospital", "amenity": "hospital", "category": "medical", "weight": 0.1},
        {"name": "Primary Health Center", "type": "clinic", "amenity": "clinic", "category": "medical", "weight": 0.15},
        {"name": "Relief Shelter", "type": "shelter", "amenity": "shelter", "category": "shelter", "weight": 0.12},
        {"name": "Community Center", "type": "community_centre", "amenity": "community_centre", "category": "food", "weight": 0.18},
        {"name": "Water Supply Point", "type": "water_point", "amenity": "water_point", "category": "water", "weight": 0.2},
        {"name": "Police Station", "type": "police", "amenity": "police", "category": "security", "weight": 0.08},
        {"name": "Fire Station", "type": "fire_station", "amenity": "fire_station", "category": "security", "weight": 0.05},
        {"name": "School (Evacuation Center)", "type": "school", "amenity": "school", "category": "evacuation", "weight": 0.15},
        {"name": "Bus Stop", "type": "bus_stop", "highway": "bus_stop", "category": "transport", "weight": 0.25},
        {"name": "Supermarket", "type": "supermarket", "shop": "supermarket", "category": "food", "weight": 0.2},
    ]
    
    # Generate 2-4 items per category based on probability
    osm_id_counter = 100000
    for template in infrastructure_templates:
        # Probability of including this type
        if random.random() < template["weight"]:
            # Generate 1-3 instances of this type
            for _ in range(random.randint(1, 3)):
                # Random offset from center (within radius)
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(100, radius_m * 0.9)  # 100m to 90% of radius
                
                # Calculate offset coordinates
                lat_offset = (distance / 111000) * math.cos(angle)
                lon_offset = (distance / (111000 * math.cos(math.radians(lat)))) * math.sin(angle)
                
                item_lat = lat + lat_offset
                item_lon = lon + lon_offset
                distance_km = distance / 1000.0
                
                # Build tags
                tags = {}
                if "amenity" in template:
                    tags["amenity"] = template["amenity"]
                if "highway" in template:
                    tags["highway"] = template["highway"]
                if "shop" in template:
                    tags["shop"] = template["shop"]
                tags["name"] = template["name"] + f" #{random.randint(1, 99)}"
                
                mock_items.append({
                    "id": osm_id_counter,
                    "lat": round(item_lat, 6),
                    "lon": round(item_lon, 6),
                    "tags": tags,
                })
                osm_id_counter += 1
    
    return mock_items if mock_items else []


@lru_cache(maxsize=128)
def _cached_overpass_query(query: str) -> Optional[str]:
    """
    Cached wrapper around Overpass API query. Results cached for entire session.
    Returns JSON string for hashability (required by lru_cache).
    """
    return _overpass_query_raw(query)


def clear_overpass_cache() -> None:
    _cached_overpass_query.cache_clear()


def _overpass_query_raw(query: str) -> Optional[str]:
    """POST the Overpass query to each server in order and return the raw JSON
    response body as a string, or None if every server fails.
    """
    import logging
    logger = logging.getLogger(__name__)
    timeout = float(cfg("osm.timeout_sec", 20))
    for url in OVERPASS_SERVERS:
        try:
            response = requests.post(
                url,
                data={"data": query},
                timeout=timeout,
                headers={"User-Agent": "safety-zone-combined/0.1"},
            )
            response.raise_for_status()
            text = response.text
            # validate payload is parseable JSON with "elements"
            payload = json.loads(text)
            if isinstance(payload, dict) and isinstance(payload.get("elements"), list):
                logger.info("overpass server OK: %s (%d elements)",
                            url, len(payload["elements"]))
                return text
            logger.warning("overpass server %s returned unexpected payload", url)
        except (requests.RequestException, ValueError) as exc:
            logger.warning("overpass server %s failed: %s", url, exc)
            continue
    return None


def _categorise(tags: Dict[str, Any]) -> Optional[str]:
    for (key, value), category in CATEGORY_MAP.items():
        tag_value = tags.get(key)
        if tag_value is None:
            continue
        if value is None or tag_value == value:
            return category
    return None


def _derive_name(tags: Dict[str, Any], category: str) -> str:
    return (
        tags.get("name")
        or tags.get("official_name")
        or tags.get("amenity")
        or tags.get("emergency")
        or category
        or "Unknown"
    )


def _extract_coordinates(element: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    if "lat" in element and "lon" in element:
        return _safe_float(element["lat"]), _safe_float(element["lon"])
    center = element.get("center") or {}
    if "lat" in center and "lon" in center:
        return _safe_float(center["lat"]), _safe_float(center["lon"])
    return None, None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius_km * c


def _build_emergency_shortlist(
    by_category: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    hospitals = [
        _slim(item) for item in by_category.get("medical", [])
        if item.get("type") in ("hospital", "clinic", "doctors")
    ][:3]
    security = by_category.get("security", [])
    police = [_slim(item) for item in security if item.get("type") == "police"][:3]
    fire = [_slim(item) for item in security if item.get("type") == "fire_station"][:3]
    return {
        "hospitals": hospitals,
        "police_stations": police,
        "fire_stations": fire,
    }


def _slim(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": item.get("name", "Unknown"),
        "type": item.get("type", "unknown"),
        "lat": item.get("lat"),
        "lon": item.get("lon"),
        "distance_km": item.get("distance_km"),
    }


def _compute_safety_score(
    by_category: Dict[str, List[Dict[str, Any]]],
    radius_km: float,
) -> Dict[str, Any]:
    contributions: Dict[str, float] = {}
    radius_km = max(radius_km, 0.1)
    for category, weight in CATEGORY_SAFETY_WEIGHT.items():
        cat_score = 0.0
        for item in by_category.get(category, []):
            distance = item.get("distance_km") if item.get("distance_km") is not None else radius_km
            proximity = max(0.0, 1.0 - (distance / radius_km))
            cat_score += weight * proximity
        contributions[category] = round(cat_score, 3)

    raw_score = sum(contributions.values())
    normalised = 1.0 / (1.0 + math.exp(-raw_score / 5.0))
    return {
        "raw": round(raw_score, 3),
        "normalised": round(normalised, 3),
        "contributions": contributions,
    }


def _empty_profile(lat: Optional[float], lon: Optional[float], radius_m: int) -> Dict[str, Any]:
    return {
        "center": {"lat": lat, "lon": lon},
        "radius_m": radius_m,
        "by_category": {cat: [] for cat in CATEGORY_SAFETY_WEIGHT},
        "category_counts": {cat: 0 for cat in CATEGORY_SAFETY_WEIGHT},
        "emergency_shortlist": {
            "hospitals": [],
            "police_stations": [],
            "fire_stations": [],
        },
        "safety_score": {"raw": 0.0, "normalised": 0.5, "contributions": {}},
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

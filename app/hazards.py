"""Hazard ingestion — weather + rule-based classification + mock alert bank.

Kept fully synchronous so the Streamlit app stays single-process and simple.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from app.config import openweather_api_key

OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


def get_weather_data(lat: float, lon: float) -> Dict[str, Any]:
    src_lat = _safe_float(lat)
    src_lon = _safe_float(lon)
    api_key = openweather_api_key()

    if src_lat is None or src_lon is None or not api_key:
        return _empty_weather(src_lat, src_lon)

    try:
        response = requests.get(
            OPENWEATHER_URL,
            params={"lat": src_lat, "lon": src_lon, "appid": api_key, "units": "metric"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return _empty_weather(src_lat, src_lon)

    main = payload.get("main", {}) if isinstance(payload, dict) else {}
    weather_list = payload.get("weather", []) if isinstance(payload, dict) else []
    wind = payload.get("wind", {}) if isinstance(payload, dict) else {}
    rain = payload.get("rain", {}) if isinstance(payload, dict) else {}

    temperature = _safe_float(main.get("temp"))
    condition = "unknown"
    if isinstance(weather_list, list) and weather_list:
        condition = str(weather_list[0].get("main", "unknown")).lower()
    wind_speed = _safe_float(wind.get("speed"))

    rainfall: Optional[float] = None
    if isinstance(rain, dict):
        rainfall = _safe_float(rain.get("1h"))
        if rainfall is None:
            rainfall = _safe_float(rain.get("3h"))

    return {
        "location": {
            "lat": src_lat,
            "lon": src_lon,
            "name": payload.get("name", "Unknown") if isinstance(payload, dict) else "Unknown",
        },
        "temperature": temperature,
        "weather_condition": condition,
        "wind_speed": wind_speed,
        "rainfall": rainfall,
        "units": {"temperature": "celsius", "wind_speed": "m/s", "rainfall": "mm"},
    }


def classify_weather_hazard(weather: Dict[str, Any]) -> Dict[str, str]:
    """Turn weather readings into a single hazard_type + severity label."""
    if not isinstance(weather, dict):
        return {"hazard_type": "none", "severity": "Low"}

    temperature = _safe_float(weather.get("temperature"))
    wind_speed = _safe_float(weather.get("wind_speed"))
    rainfall = _safe_float(weather.get("rainfall"))
    condition = str(weather.get("weather_condition", "")).lower()

    hazard_type = "none"
    severity = "Low"

    if rainfall is not None and rainfall >= 30:
        hazard_type, severity = "flood risk", "High"
    elif rainfall is not None and rainfall >= 15:
        hazard_type, severity = "flood risk", "Medium"
    elif "rain" in condition and rainfall is None:
        hazard_type, severity = "flood risk", "Medium"

    if wind_speed is not None and wind_speed >= 20:
        hazard_type, severity = "storm risk", "High"
    elif wind_speed is not None and wind_speed >= 12 and severity == "Low":
        hazard_type, severity = "storm risk", "Medium"

    if temperature is not None and temperature >= 40:
        hazard_type, severity = "heatwave risk", "High"
    elif temperature is not None and temperature >= 35 and severity == "Low":
        hazard_type, severity = "heatwave risk", "Medium"

    return {"hazard_type": hazard_type, "severity": severity}


def _mock_alert_bank() -> List[Dict[str, str]]:
    """Small curated list of SACHET-style alerts used when live feeds are off."""
    return [
        {
            "type": "flood risk",
            "severity": "High",
            "location": "Low-lying settlements near riverbanks",
            "description": "Heavy rainfall may cause localised flooding and water logging.",
        },
        {
            "type": "storm risk",
            "severity": "Medium",
            "location": "Coastal and exposed areas",
            "description": "Strong winds may disrupt travel and damage temporary structures.",
        },
        {
            "type": "heatwave risk",
            "severity": "Medium",
            "location": "Urban heat-prone neighbourhoods",
            "description": "High temperatures may increase dehydration and heat stress.",
        },
        {
            "type": "landslide risk",
            "severity": "Medium",
            "location": "Hillside villages and cut slopes",
            "description": "Saturated soil after prolonged rain increases slope failure risk.",
        },
    ]


def _mock_history_bank() -> List[Dict[str, str]]:
    """Tiny historical bulletin bank used for grounded evidence snippets."""
    return [
        {
            "type": "flood",
            "severity": "High",
            "location": "Panchganga basin, Kolhapur",
            "description": "2019 and 2021 monsoon flooding displaced thousands and inundated low-lying settlements.",
        },
        {
            "type": "cyclone",
            "severity": "High",
            "location": "Konkan coast, Maharashtra",
            "description": "Cyclone Tauktae (2021) caused evacuation in Ratnagiri and Sindhudurg coastal villages.",
        },
        {
            "type": "landslide",
            "severity": "Medium",
            "location": "Western Ghats, Raigad",
            "description": "Recurrent landslides during peak monsoon in Irshalwadi and Talai villages.",
        },
        {
            "type": "heatwave",
            "severity": "Medium",
            "location": "Vidarbha region, Maharashtra",
            "description": "Heatwave events in summer lead to ORS distribution and cooling shelters in Nagpur/Akola.",
        },
    ]


def get_hazard_context(lat: float, lon: float, query: str) -> Dict[str, Any]:
    weather = get_weather_data(lat, lon)
    hazard = classify_weather_hazard(weather)
    alerts = _filter_by_relevance(_mock_alert_bank(), query, hazard)
    history = _filter_by_relevance(_mock_history_bank(), query, hazard)
    return {
        "weather": weather,
        "hazard": hazard,
        "alerts": alerts,
        "history": history,
    }


def _filter_by_relevance(
    records: List[Dict[str, Any]],
    query: str,
    hazard: Dict[str, str],
) -> List[Dict[str, Any]]:
    if not records:
        return []

    hazard_type = str(hazard.get("hazard_type", "")).lower()
    keywords = [word for word in str(query or "").lower().split() if len(word) > 2]

    matched: List[Dict[str, Any]] = []
    for record in records:
        blob = " ".join(str(v).lower() for v in record.values())
        if hazard_type and hazard_type.split()[0] in blob:
            matched.append(record)
            continue
        if keywords and any(keyword in blob for keyword in keywords):
            matched.append(record)

    return (matched or records)[:5]


def _empty_weather(lat: Optional[float], lon: Optional[float]) -> Dict[str, Any]:
    return {
        "location": {"lat": lat, "lon": lon, "name": "Unknown"},
        "temperature": None,
        "weather_condition": "unknown",
        "wind_speed": None,
        "rainfall": None,
        "units": {"temperature": "celsius", "wind_speed": "m/s", "rainfall": "mm"},
    }


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

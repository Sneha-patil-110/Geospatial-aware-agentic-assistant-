"""Safety Zone Combined — unified Streamlit UI.

Combines features from:
  * Enigma_AI_assistant (hazard + RAG + NVIDIA reasoning orchestration)
  * safety-zone-agentic-assistant-main (rich OSM profile + grounded docs RAG)

Two focus tabs as requested:
  * Emergency Infrastructure panel — categorised OSM lists + safety score + map.
  * RAG Q&A chat — conversational queries over the combined docs + camps corpus.

Plus a shared "Analyse location" action that runs the full pipeline.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import folium
import streamlit as st
from streamlit_folium import st_folium
from dotenv import load_dotenv

from app.assistant import answer_question, run_assistant
from app import rag, config

# Force reload environment variables and clear all caches
load_dotenv(override=True)
config.clear_config_cache()
rag.clear_embedding_cache()

st.set_page_config(
    page_title="Safety Zone Combined",
    page_icon="🛟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Verify APIs are available at startup
def _verify_apis() -> None:
    from app.config import nvidia_embedding_api_key, nvidia_reasoning_api_key
    emb_key = nvidia_embedding_api_key()
    llm_key = nvidia_reasoning_api_key()
    if not emb_key or not llm_key:
        st.error("⚠️ API keys not found in .env file. LLM and embedding services will not work.")
        st.stop()

_verify_apis()

RISK_COLORS = {
    "LOW": "#1FA64A",
    "MEDIUM": "#E7B416",
    "HIGH": "#D93A3A",
    "UNKNOWN": "#6E7781",
}

RISK_EMOJI = {
    "LOW": "🟢",
    "MEDIUM": "🟡",
    "HIGH": "🔴",
    "UNKNOWN": "⚪",
}

PRESETS: Dict[str, Tuple[float, float, str, str, str]] = {
    "Pune": (18.5204, 73.8567, "Pune", "Maharashtra", "Evacuation safety and relief camps near me?"),
    "Mumbai": (19.0760, 72.8777, "Mumbai", "Maharashtra", "Cyclone evacuation guidance for coastal wards?"),
    "Raigad": (18.2485, 73.1305, "Raigad", "Maharashtra", "Landslide risk and nearest shelters?"),
    "Kolhapur": (16.7050, 74.2433, "Kolhapur", "Maharashtra", "Panchganga flood risk and relief camps?"),
    "Nagpur": (21.1458, 79.0882, "Nagpur", "Maharashtra", "Heatwave advisory and cooling shelters?"),
}

CATEGORY_LABELS = {
    "medical": "🏥 Medical",
    "shelter": "🏕️ Shelter",
    "water": "💧 Water",
    "food": "🥫 Food",
    "security": "🚓 Security",
    "evacuation": "🎓 Evacuation",
    "transport": "🚌 Transport",
}


# ─── State helpers ────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "lat": 18.5204,
        "lon": 73.8567,
        "district": "Pune",
        "state": "Maharashtra",
        "query": "Evacuation safety and relief camps near me?",
        "radius_m": 5000,
        "result": None,
        "chat_history": [],  # list of {role, content, citations?}
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _apply_preset(name: str) -> None:
    lat, lon, district, state, query = PRESETS[name]
    st.session_state.lat = lat
    st.session_state.lon = lon
    st.session_state.district = district
    st.session_state.state = state
    st.session_state.query = query


# ─── Rendering helpers ────────────────────────────────────────────────────────

def _risk_level(result: Dict[str, Any]) -> str:
    analysis = result.get("risk_analysis") or {}
    if isinstance(analysis, dict):
        level = str(analysis.get("risk_level", "")).upper()
        if level in RISK_COLORS:
            return level
    scoring_label = str((result.get("risk_scoring") or {}).get("label", "")).upper()
    if scoring_label in ("SAFE",):
        return "LOW"
    if scoring_label in ("MODERATE",):
        return "MEDIUM"
    if scoring_label in ("UNSAFE",):
        return "HIGH"
    return "UNKNOWN"


def _render_headline_badges(result: Dict[str, Any]) -> None:
    level = _risk_level(result)
    colour = RISK_COLORS[level]

    infra = result.get("infra") or {}
    counts = infra.get("category_counts", {}) if isinstance(infra, dict) else {}
    total_items = sum(int(v) for v in counts.values() if isinstance(v, (int, float)))

    scoring = result.get("risk_scoring") or {}
    score = scoring.get("score")
    score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"

    badges_html = f"""
    <div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;'>
        <span style='background:{colour};color:white;padding:6px 12px;border-radius:999px;font-weight:700;'>
            {RISK_EMOJI[level]} Risk: {level}
        </span>
        <span style='background:#1f2937;color:white;padding:6px 12px;border-radius:999px;font-weight:700;'>
            Infra items: {total_items}
        </span>
        <span style='background:#334155;color:white;padding:6px 12px;border-radius:999px;font-weight:700;'>
            District score: {score_text} ({scoring.get('label', 'n/a')})
        </span>
    </div>
    """
    st.markdown(badges_html, unsafe_allow_html=True)


def _build_map(result: Dict[str, Any]) -> folium.Map:
    centre = [st.session_state.lat, st.session_state.lon]
    m = folium.Map(location=centre, zoom_start=12, tiles="OpenStreetMap")

    level = _risk_level(result) if result else "UNKNOWN"
    colour = RISK_COLORS[level]

    folium.Marker(
        location=centre,
        tooltip="Selected location",
        popup=f"Lat: {centre[0]:.4f}, Lon: {centre[1]:.4f}",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    folium.Circle(
        location=centre,
        radius=st.session_state.radius_m,
        color=colour,
        fill=True,
        fill_opacity=0.12,
        weight=2,
    ).add_to(m)

    infra = (result or {}).get("infra") or {}
    by_category = infra.get("by_category", {}) if isinstance(infra, dict) else {}
    for category, items in by_category.items():
        for item in items:
            if item.get("lat") is None or item.get("lon") is None:
                continue
            popup = (
                f"<b>{item.get('name', 'Unknown')}</b><br>"
                f"Category: {category}<br>"
                f"Type: {item.get('type', 'unknown')}<br>"
                f"Distance: {item.get('distance_km', 'n/a')} km"
            )
            folium.CircleMarker(
                location=[item["lat"], item["lon"]],
                radius=5,
                color=_category_colour(category),
                fill=True,
                fill_opacity=0.85,
                popup=popup,
                tooltip=item.get("name", category),
            ).add_to(m)

    camps = (result or {}).get("nearby_camps") or []
    for camp in camps:
        meta = camp.get("metadata", {}) if isinstance(camp, dict) else {}
        camp_lat = meta.get("lat")
        camp_lon = meta.get("lon")
        if camp_lat is None or camp_lon is None:
            continue
        folium.Marker(
            location=[camp_lat, camp_lon],
            tooltip=camp.get("title", "Camp"),
            popup=f"<b>{camp.get('title', 'Camp')}</b><br>{camp.get('text', '')[:200]}",
            icon=folium.Icon(color="green", icon="home"),
        ).add_to(m)

    return m


def _category_colour(category: str) -> str:
    palette = {
        "medical": "#d62728",
        "shelter": "#2ca02c",
        "water": "#1f77b4",
        "food": "#ff7f0e",
        "security": "#9467bd",
        "evacuation": "#17becf",
        "transport": "#7f7f7f",
    }
    return palette.get(category, "#333333")


def _render_infra_panel(result: Dict[str, Any]) -> None:
    infra = result.get("infra") or {}
    if not isinstance(infra, dict) or not infra:
        st.info("Run analysis to populate infrastructure data.")
        return

    counts = infra.get("category_counts", {}) or {}
    
    # Show category distribution chart
    if counts and any(counts.values()):
        st.bar_chart(counts, use_container_width=True)
    
    # Show safety score metric
    safety_score = infra.get("safety_score", {}) or {}
    normalised = safety_score.get("normalised")
    if isinstance(normalised, (int, float)):
        st.metric("Area safety score", f"{normalised:.2f}", help="0 = sparse infra, 1 = dense coverage")

    # Show emergency shortlist prominently at the top
    emergency_shortlist = infra.get("emergency_shortlist", {}) or {}
    if emergency_shortlist:
        st.markdown("#### Emergency Shortlist")
        shortlist_cols = st.columns(3)
        
        hospitals = emergency_shortlist.get("hospitals", [])
        with shortlist_cols[0]:
            st.metric("Hospitals", len(hospitals))
            if hospitals:
                for h in hospitals:
                    dist = h.get("distance_km", "N/A")
                    st.caption(f"📍 {h.get('name', 'Unknown')} - {dist}km")
        
        police = emergency_shortlist.get("police_stations", [])
        with shortlist_cols[1]:
            st.metric("Police Stations", len(police))
            if police:
                for p in police:
                    dist = p.get("distance_km", "N/A")
                    st.caption(f"👮 {p.get('name', 'Unknown')} - {dist}km")
        
        fire = emergency_shortlist.get("fire_stations", [])
        with shortlist_cols[2]:
            st.metric("Fire Stations", len(fire))
            if fire:
                for f in fire:
                    dist = f.get("distance_km", "N/A")
                    st.caption(f"🚒 {f.get('name', 'Unknown')} - {dist}km")
        
        st.markdown("---")

    # Show full infrastructure by category
    by_category: Dict[str, List[Dict[str, Any]]] = infra.get("by_category", {}) or {}
    
    # Filter out empty categories
    non_empty = {cat: items for cat, items in by_category.items() if items}
    
    if non_empty:
        st.markdown("#### All Infrastructure by Category")
        for category, items in sorted(non_empty.items()):
            label = CATEGORY_LABELS.get(category, category.title())
            with st.expander(f"{label} — {len(items)} found", expanded=(category == "medical")):
                # Create dataframe with more details
                rows = [
                    {
                        "Name": item.get("name", "Unknown"),
                        "Type": item.get("type", "unknown"),
                        "Distance (km)": round(float(item.get("distance_km", 0)), 2) if item.get("distance_km") else "N/A",
                        "Lat": item.get("lat", "N/A"),
                        "Lon": item.get("lon", "N/A"),
                    }
                    for item in sorted(items, key=lambda x: x.get("distance_km") or float("inf"))
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No infrastructure items found in the selected area.")


def _render_risk_analysis(result: Dict[str, Any]) -> None:
    analysis = result.get("risk_analysis") or {}
    level = str(analysis.get("risk_level", "Unknown")).capitalize()
    explanation = analysis.get("explanation", "No explanation generated.")
    action = analysis.get("recommended_action", "No recommended action generated.")
    used_llm = analysis.get("used_llm", False)

    st.markdown(f"**Risk Level:** {level}")
    st.markdown(f"**Explanation:** {explanation}")
    st.markdown(f"**Recommended Action:** {action}")
    
    # Show LLM status prominently
    if used_llm:
        st.success("✓ Generated with NVIDIA Nemotron LLM")
    else:
        st.warning("⚠️ Using rule-based fallback (LLM not available or failed)")
        st.caption("This means the analysis is based on rules, not AI reasoning. Check .env API keys if this persists.")


def _render_chat_tab() -> None:
    st.markdown(
        "Ask a question grounded in the combined corpus: NDMA guidelines, the landslide "
        "atlas summary, flood inventory notes, relief camp protocols, and the refugee/"
        "relief camp directory."
    )

    # Display existing chat history first
    for message in st.session_state.chat_history[-12:]:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                citations = message.get("citations", [])
                if citations:
                    st.caption("Sources: " + ", ".join(f"[{c['title']}]" for c in citations))
                st.caption(
                    "Answered with NVIDIA Nemotron." if message.get("used_llm")
                    else "Rule-based fallback (no NVIDIA key)."
                )
                docs = message.get("retrieved_docs", [])
                if docs:
                    with st.expander("Show retrieved documents"):
                        for doc in docs:
                            st.markdown(f"**[{doc.get('source', 'unknown')}] {doc.get('title', '')}**")
                            st.caption(f"score ≈ {doc.get('score', 'n/a')}")
                            st.write((doc.get("text") or "")[:500] + ("..." if len(doc.get("text") or "") > 500 else ""))

    # Input form with proper clearing
    with st.form(key="chat_form", clear_on_submit=True):
        chat_query = st.text_area(
            "Your question",
            value="What does NDMA recommend for flood evacuation shelters?",
            height=90,
            key="chat_input",
        )
        submitted = st.form_submit_button("Ask", type="primary")

    # Process new submission
    if submitted and chat_query.strip():
        # Immediately show user message
        with st.chat_message("user"):
            st.write(chat_query.strip())
        
        # Get and display assistant response
        with st.spinner("Retrieving and reasoning..."):
            reply = answer_question(chat_query.strip(), top_k_docs=6)
        
        # Store in history
        st.session_state.chat_history.append({"role": "user", "content": chat_query.strip()})
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": reply["answer"],
                "citations": reply.get("citations", []),
                "used_llm": reply.get("used_llm", False),
                "retrieved_docs": reply.get("retrieved_docs", []),
            }
        )
        
        # Immediately display assistant response
        with st.chat_message("assistant"):
            st.write(reply["answer"])
            citations = reply.get("citations", [])
            if citations:
                st.caption("Sources: " + ", ".join(f"[{c['title']}]" for c in citations))
            st.caption(
                "Answered with NVIDIA Nemotron." if reply.get("used_llm")
                else "Rule-based fallback (no NVIDIA key)."
            )
            docs = reply.get("retrieved_docs", [])
            if docs:
                with st.expander("Show retrieved documents"):
                    for doc in docs:
                        st.markdown(f"**[{doc.get('source', 'unknown')}] {doc.get('title', '')}**")
                        st.caption(f"score ≈ {doc.get('score', 'n/a')}")
                        st.write((doc.get("text") or "")[:500] + ("..." if len(doc.get("text") or "") > 500 else ""))
        
        st.rerun()

    if st.session_state.chat_history and st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()


# ─── Main layout ──────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()

    st.markdown("## 🛟 Safety Zone Combined — Civilian Safety Zone Monitor")
    st.caption("Hazard intelligence + emergency infra + grounded RAG chat, in a single Streamlit app.")

    with st.sidebar:
        st.markdown("### 📍 Quick presets")
        cols = st.columns(2)
        for idx, preset in enumerate(PRESETS.keys()):
            col = cols[idx % 2]
            if col.button(preset, width="stretch", key=f"preset_{preset}"):
                _apply_preset(preset)

        st.markdown("---")
        st.markdown("### Inputs")
        st.session_state.lat = st.number_input(
            "Latitude", min_value=-90.0, max_value=90.0,
            value=float(st.session_state.lat), step=0.0001, format="%.4f",
        )
        st.session_state.lon = st.number_input(
            "Longitude", min_value=-180.0, max_value=180.0,
            value=float(st.session_state.lon), step=0.0001, format="%.4f",
        )
        st.session_state.district = st.text_input("District (for priors)", value=st.session_state.district)
        st.session_state.state = st.text_input("State", value=st.session_state.state)
        st.session_state.query = st.text_input("Query", value=st.session_state.query)
        st.session_state.radius_m = st.slider(
            "Search radius (m)", min_value=1000, max_value=20000,
            value=int(st.session_state.radius_m), step=500,
        )

        st.markdown("---")
        if st.button("🔄 Analyse location", type="primary", width="stretch"):
            with st.spinner("Running full pipeline (OSM + hazards + RAG + reasoning)..."):
                try:
                    st.session_state.result = run_assistant(
                        lat=float(st.session_state.lat),
                        lon=float(st.session_state.lon),
                        query=st.session_state.query,
                        district=st.session_state.district or None,
                        state=st.session_state.state or None,
                        radius_m=int(st.session_state.radius_m),
                    )
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

    infra_tab, chat_tab, details_tab = st.tabs(
        ["🚑 Emergency Infrastructure", "💬 RAG Q&A", "🧾 Full output"]
    )

    result = st.session_state.result or {}

    with infra_tab:
        left, right = st.columns([1.2, 1.0])
        with left:
            st.markdown("#### 🗺️ Live map")
            map_obj = _build_map(result)
            st_folium(map_obj, height=520, use_container_width=True, returned_objects=[])

            st.markdown("#### 📌 Risk summary")
            if result:
                _render_headline_badges(result)
                _render_risk_analysis(result)

                infra = result.get("infra") or {}
                infra_err = infra.get("error") if isinstance(infra, dict) else None
                if infra_err:
                    st.error(
                        f"🚧 OSM fetch issue: **{infra_err}**. "
                        "If this persists, run `python diagnose.py` to check Overpass reachability."
                    )

                warnings = result.get("warnings") or []
                if warnings:
                    st.warning(f"⚠️ Pipeline warnings ({len(warnings)}):")
                    for warning in warnings:
                        st.caption(f"• {warning}")
            else:
                st.info("Pick a location and click **Analyse location** in the sidebar.")

        with right:
            st.markdown("#### 🚑 Nearby infrastructure")
            _render_infra_panel(result)

    with chat_tab:
        _render_chat_tab()

    with details_tab:
        if not result:
            st.info("No analysis has run yet.")
        else:
            st.markdown("#### Retrieved documents")
            docs = result.get("retrieved_docs") or []
            if docs:
                st.dataframe(
                    [
                        {
                            "Title": doc.get("title", ""),
                            "Source": doc.get("source", ""),
                            "Kind": (doc.get("metadata") or {}).get("kind", "document"),
                            "Score": doc.get("score"),
                        }
                        for doc in docs
                    ],
                    width="stretch",
                    hide_index=True,
                )
            st.markdown("#### Raw pipeline output")
            st.code(json.dumps(result, indent=2, default=str)[:20000], language="json")


if __name__ == "__main__":
    main()

# Safety Zone Combined

A single-process Streamlit agentic assistant that merges two earlier projects:

- `Enigma_AI_assistant` — OSM emergency infra, hazard ingestion, refugee-camp RAG with NVIDIA embeddings, NVIDIA Nemotron reasoning, Streamlit live demo.
- `safety-zone-agentic-assistant-main` — FastAPI agentic loop, hybrid BM25+FAISS RAG over NDMA SOPs / landslide atlas / flood inventory, deterministic district scoring with GeoJSON priors.

This folder combines the best of both into one runnable Streamlit app — no FastAPI backend required.

## Feature summary

- **Emergency Infrastructure panel** — OSM Overpass with 7 categories (medical, shelter, water, food, security, evacuation, transport), distance-sorted lists, bar chart of counts, composite safety score, and a live map with markers.
- **RAG Q&A chat** — grounded question answering over the combined corpus (NDMA SOPs, landslide atlas summary, flood inventory summary, relief camp protocols, forest fire monitoring, and India-focused relief camp records).
- **Full pipeline "Analyse location"** — OSM infra + weather + hazard classification + alerts/history + RAG retrieval + deterministic district risk score + NVIDIA Nemotron risk analysis, all in one click.
- **Robust fallbacks** — every step works without internet or NVIDIA keys (deterministic embeddings, rule-based risk reasoning, mock alert bank). When keys are present the app uses NVIDIA `nv-embedqa-e5-v5` and `nemotron-3-nano-30b-a3b`.

## Project layout

```
safety-zone-combined/
├── app/
│   ├── __init__.py
│   ├── config.py          # dotenv + config.yaml loader
│   ├── nearby_infra.py    # OSM Overpass, rich categories + emergency shortlist
│   ├── hazards.py         # weather, classification, mock alerts + history
│   ├── rag.py             # unified NVIDIA-embeddings FAISS over docs + camps
│   ├── risk_scoring.py    # deterministic district scoring (flood + landslide priors)
│   ├── reasoning.py       # Nemotron risk + chat answers, both with fallbacks
│   └── assistant.py       # top-level orchestrator
├── data/
│   ├── docs/              # NDMA SOPs, atlas/inventory summaries (.txt + .meta.json)
│   ├── geo/               # flood_inventory.geojson, landslide_atlas.geojson
│   └── refugee_camps.json # India-focused relief camp directory
├── config.yaml
├── requirements.txt
├── .env.example
├── streamlit_app.py       # combined Streamlit UI
└── README.md
```

## Getting started

```powershell
# From this folder
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Copy the env template and fill your keys (optional)
copy .env.example .env
# Edit .env and set NVIDIA_API_KEY=... and OPENWEATHER_API_KEY=...

streamlit run streamlit_app.py
```

Open <http://localhost:8501> in your browser.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `NVIDIA_API_KEY` | Shared NVIDIA NIM key (used for embeddings + reasoning unless overridden). |
| `NVIDIA_EMBEDDING_API_KEY` | Optional override for embeddings (`nv-embedqa-e5-v5`). |
| `NVIDIA_REASONING_API_KEY` | Optional override for reasoning (`nemotron-3-nano-30b-a3b`). |
| `OPENWEATHER_API_KEY` | Optional OpenWeatherMap key for live weather-based hazard classification. |

When a key is missing the corresponding subsystem degrades gracefully — the app still works end-to-end with deterministic fallbacks.

## Presets

The sidebar has one-click presets for Pune, Mumbai, Raigad, Kolhapur, and Nagpur with sensible starting queries.

## Notes on combining the two projects

- The agentic ReAct loop from `safety-zone` was simplified into a single synchronous orchestrator (`app/assistant.py`) so the app runs in one Streamlit process with no FastAPI backend.
- The hybrid BM25+FAISS retriever was replaced with NVIDIA embeddings + FAISS over a unified corpus (docs + camps) — same grounding, simpler runtime.
- Overpass categories, safety score, and the district-level risk scoring were ported verbatim (minus the async httpx client and the event-store SQLite dependency).
- All NVIDIA API calls live behind lazy, fail-soft wrappers so the Streamlit app never crashes because of a missing or flaky key.

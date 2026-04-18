"""End-to-end scenario tests for the combined assistant.

Mirrors the Pune / Mumbai / Rural scenarios from Enigma's original
test_assistant.py but targets the combined pipeline (`app.assistant`).

Usage:
    python test_assistant.py                  # run all scenarios
    python test_assistant.py --chat-only      # RAG Q&A only (no OSM call)
    python test_assistant.py --no-network     # skip scenarios that hit Overpass
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from app.assistant import answer_question, run_assistant

SCENARIOS = [
    {
        "name": "Pune urban",
        "lat": 18.5204,
        "lon": 73.8567,
        "district": "Pune",
        "state": "Maharashtra",
        "query": "Evacuation safety and relief camps near me?",
        "radius_m": 3000,
    },
    {
        "name": "Mumbai coastal",
        "lat": 19.0760,
        "lon": 72.8777,
        "district": "Mumbai",
        "state": "Maharashtra",
        "query": "Cyclone evacuation guidance for coastal wards?",
        "radius_m": 4000,
    },
    {
        "name": "Rural Maharashtra",
        "lat": 19.2183,
        "lon": 72.9781,
        "district": "Thane",
        "state": "Maharashtra",
        "query": "Flood risk and nearest relief infrastructure?",
        "radius_m": 5000,
    },
]

CHAT_QUERIES = [
    "What does NDMA recommend for flood evacuation shelters?",
    "What minimum standards apply to relief camps?",
    "How are landslide-prone districts classified in India?",
]


def _print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _summarise_result(result: Dict[str, Any]) -> None:
    loc = result.get("location", {})
    risk_analysis = result.get("risk_analysis", {}) or {}
    scoring = result.get("risk_scoring", {}) or {}
    infra = result.get("infra", {}) or {}

    print(f"  Location:     lat={loc.get('lat')} lon={loc.get('lon')} "
          f"district={loc.get('district')} state={loc.get('state')}")
    print(f"  Risk Level:   {risk_analysis.get('risk_level')}  "
          f"(used_llm={risk_analysis.get('used_llm')})")
    print(f"  Explanation:  {risk_analysis.get('explanation', '')[:200]}")
    print(f"  Action:       {risk_analysis.get('recommended_action', '')[:200]}")
    print(f"  District score: {scoring.get('score')} ({scoring.get('label')})")

    counts = infra.get("category_counts", {}) if isinstance(infra, dict) else {}
    total = sum(int(v) for v in counts.values() if isinstance(v, (int, float)))
    print(f"  Infra items:  total={total}  counts={counts}")

    docs = result.get("retrieved_docs", [])
    print(f"  Retrieved docs: {len(docs)}")
    for doc in docs[:3]:
        print(f"    - [{doc.get('source')}] {doc.get('title')}  score={doc.get('score')}")

    warnings = result.get("warnings", [])
    if warnings:
        print("  Warnings:")
        for warning in warnings:
            print(f"    - {warning}")


def run_full_scenarios(no_network: bool) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []
    for scenario in SCENARIOS:
        _print_section(f"SCENARIO: {scenario['name']}")
        if no_network:
            print("  (network scenarios skipped; using chat-only mode)")
            continue
        try:
            result = run_assistant(
                lat=scenario["lat"],
                lon=scenario["lon"],
                query=scenario["query"],
                district=scenario["district"],
                state=scenario["state"],
                radius_m=scenario["radius_m"],
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue
        _summarise_result(result)
        all_results.append(result)
    return all_results


def run_chat_queries() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for query in CHAT_QUERIES:
        _print_section(f"CHAT: {query}")
        reply = answer_question(query, top_k_docs=4)
        print(f"  used_llm={reply.get('used_llm')}")
        citations = reply.get("citations", [])
        if citations:
            print("  citations: " + ", ".join(f"[{c['title']}]" for c in citations))
        print("  answer (first 400 chars):")
        answer = (reply.get("answer") or "").strip()
        print("    " + answer[:400].replace("\n", "\n    "))
        results.append(reply)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-only", action="store_true", help="Only run chat queries.")
    parser.add_argument("--no-network", action="store_true", help="Skip scenarios that hit Overpass.")
    parser.add_argument("--save", type=str, default=None, help="Optional path to dump JSON results.")
    args = parser.parse_args()

    full_results: List[Dict[str, Any]] = []
    chat_results: List[Dict[str, Any]] = []

    if not args.chat_only:
        full_results = run_full_scenarios(no_network=args.no_network)

    chat_results = run_chat_queries()

    if args.save:
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(
                {"scenarios": full_results, "chat": chat_results},
                fh,
                indent=2,
                default=str,
            )
        print(f"\nSaved results to {args.save}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

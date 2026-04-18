"""Diagnostic script — run this if the app shows 0 infrastructure items
or the LLM keeps falling back. It prints exactly what works and what
doesn't, so we can pinpoint the failure.

    python diagnose.py
"""
from __future__ import annotations

import json
import sys
import time

import requests

from app.config import nvidia_embedding_api_key, nvidia_reasoning_api_key
from app.nearby_infra import OVERPASS_SERVERS, get_infra_profile


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def check_overpass_servers() -> None:
    section("1. Overpass API reachability")
    simple_query = """
    [out:json][timeout:10];
    node["amenity"="hospital"](around:500,18.5204,73.8567);
    out body;
    """.strip()
    for url in OVERPASS_SERVERS:
        t0 = time.time()
        try:
            r = requests.post(url, data={"data": simple_query},
                              timeout=15,
                              headers={"User-Agent": "safety-zone-diagnose/1.0"})
            dt = (time.time() - t0) * 1000
            if r.status_code == 200:
                count = len(r.json().get("elements", []))
                print(f"  OK  {url}  ({dt:.0f} ms, {count} hospitals near Pune)")
            else:
                print(f"  HTTP {r.status_code}  {url}  ({dt:.0f} ms)")
        except Exception as exc:
            dt = (time.time() - t0) * 1000
            print(f"  FAIL {url}  ({dt:.0f} ms)  {exc.__class__.__name__}: {exc}")


def check_infra_profile() -> None:
    section("2. Full infra profile — Pune centre, 5 km radius")
    profile = get_infra_profile(18.5204, 73.8567, radius_m=5000)
    print(f"  error           : {profile.get('error')}")
    counts = profile.get("category_counts", {})
    total = sum(counts.values())
    print(f"  total items     : {total}")
    for cat, n in counts.items():
        print(f"    {cat:12s}: {n}")
    if total == 0:
        print("  >>> ZERO items. Most likely Overpass isn't reachable from your network.")
    shortlist = profile.get("emergency_shortlist", {})
    print(f"  hospitals       : {len(shortlist.get('hospitals', []))}")
    print(f"  police stations : {len(shortlist.get('police_stations', []))}")
    print(f"  fire stations   : {len(shortlist.get('fire_stations', []))}")


def check_embedding() -> None:
    section("3. NVIDIA embedding endpoint")
    key = nvidia_embedding_api_key()
    if not key:
        print("  NO KEY — set NVIDIA_EMBEDDING_API_KEY in .env")
        return
    print(f"  key prefix      : {key[:10]}...")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1", timeout=30)
        r = client.embeddings.create(
            input=["Evacuation safety"],
            model="nvidia/nv-embedqa-e5-v5",
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"},
        )
        dim = len(r.data[0].embedding)
        print(f"  OK              : dim={dim}")
    except Exception as exc:
        print(f"  FAIL            : {exc.__class__.__name__}: {exc}")


def check_llm_raw() -> None:
    section("4. NVIDIA Nemotron raw output (what actually comes back)")
    key = nvidia_reasoning_api_key()
    if not key:
        print("  NO KEY — set NVIDIA_REASONING_API_KEY in .env")
        return
    print(f"  key prefix      : {key[:10]}...")

    prompt = (
        "Return ONLY a JSON object with keys risk_level (Low|Medium|High), "
        "explanation (string), recommended_action (string). "
        "Scenario: Pune, heavy rainfall 45 mm, flood alert active, "
        "hospital and shelter within 2 km."
    )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1", timeout=90)
        r = client.chat.completions.create(
            model="nvidia/nemotron-3-nano-30b-a3b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        text = r.choices[0].message.content or ""
        print(f"  OK              : length={len(text)} chars")
        print("  --- raw content ---")
        print(text[:3000])
        print("  --- end raw ---")
        # try to parse
        try:
            start = text.find("{")
            end = text.rfind("}")
            parsed = json.loads(text[start:end + 1]) if start != -1 and end > start else None
            print(f"  parse JSON      : {parsed is not None}")
            if parsed:
                print(f"  parsed keys     : {list(parsed.keys())}")
        except Exception as exc:
            print(f"  JSON parse FAIL : {exc}")
    except Exception as exc:
        print(f"  FAIL            : {exc.__class__.__name__}: {exc}")


def main() -> int:
    check_overpass_servers()
    check_infra_profile()
    check_embedding()
    check_llm_raw()
    print("\nDone. Paste this whole output into chat if anything failed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Quick smoke test for NVIDIA API keys.

Run this once after filling in .env to confirm both the embedding and
reasoning endpoints are reachable from your machine.

    python test_keys.py
"""
from __future__ import annotations

import sys

from openai import OpenAI

from app.config import nvidia_embedding_api_key, nvidia_reasoning_api_key


def test_embeddings() -> bool:
    key = nvidia_embedding_api_key()
    if not key:
        print("[embed] NO KEY set in .env — skipping")
        return False
    print(f"[embed] using key prefix={key[:10]}... len={len(key)}")
    try:
        client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1", timeout=30.0)
        resp = client.embeddings.create(
            input=["What is the capital of France?"],
            model="nvidia/nv-embedqa-e5-v5",
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"},
        )
        vec = resp.data[0].embedding
        print(f"[embed] OK — dim={len(vec)} first5={[round(v,4) for v in vec[:5]]}")
        return True
    except Exception as exc:
        print(f"[embed] FAILED: {exc}")
        return False


def test_reasoning() -> bool:
    key = nvidia_reasoning_api_key()
    if not key:
        print("[llm] NO KEY set in .env — skipping")
        return False
    print(f"[llm]   using key prefix={key[:10]}... len={len(key)}")
    try:
        client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1", timeout=60.0)
        resp = client.chat.completions.create(
            model="nvidia/nemotron-3-nano-30b-a3b",
            messages=[{"role": "user", "content": "Reply with exactly: safety zone alive"}],
            temperature=0.2,
            max_tokens=30,
        )
        content = (resp.choices[0].message.content or "").strip()
        print(f"[llm]   OK — response: {content[:200]!r}")
        return True
    except Exception as exc:
        print(f"[llm]   FAILED: {exc}")
        return False


def main() -> int:
    print("=" * 60)
    print("NVIDIA NIM key smoke test")
    print("=" * 60)
    e_ok = test_embeddings()
    r_ok = test_reasoning()
    print("-" * 60)
    print(f"Summary: embeddings={'OK' if e_ok else 'FAIL'}   reasoning={'OK' if r_ok else 'FAIL'}")
    return 0 if (e_ok and r_ok) else 1


if __name__ == "__main__":
    sys.exit(main())

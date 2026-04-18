"""Config loader — merges config.yaml + environment variables."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# Project root = one level above this file's `app/` package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env once, searching from the project root.
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


@lru_cache(maxsize=1)
def _raw_yaml_config() -> Dict[str, Any]:
    cfg_path = PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def cfg(key_path: str, default: Any = None) -> Any:
    """Dot-path accessor. Example: cfg('reasoning.model')."""
    node: Any = _raw_yaml_config()
    for part in key_path.split("."):
        if not isinstance(node, dict):
            return default
        if part not in node:
            return default
        node = node[part]
    return node if node is not None else default


def resolve_path(relative: str) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(relative)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def nvidia_embedding_api_key() -> str:
    """Prefer the embedding-specific key, fall back to shared NVIDIA_API_KEY."""
    return (
        os.getenv("NVIDIA_EMBEDDING_API_KEY", "").strip()
        or os.getenv("NVIDIA_API_KEY", "").strip()
    )


def nvidia_reasoning_api_key() -> str:
    """Prefer the reasoning-specific key, fall back to shared NVIDIA_API_KEY."""
    return (
        os.getenv("NVIDIA_REASONING_API_KEY", "").strip()
        or os.getenv("NVIDIA_API_KEY", "").strip()
    )


def openweather_api_key() -> str:
    return os.getenv("OPENWEATHER_API_KEY", "").strip()


def clear_config_cache() -> None:
    """Force re-read of config.yaml on next access. Useful after env reload."""
    _raw_yaml_config.cache_clear()

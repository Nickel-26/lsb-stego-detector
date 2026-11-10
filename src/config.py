"""Configuration loading with simple deep-merge of YAML files."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Repository root = parent of this file's directory.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, **overrides: Any) -> dict[str, Any]:
    """Load the default config, deep-merge an optional user config and overrides.

    Dotted overrides are supported, e.g. ``load_config(cnn__epochs=10)``.
    """
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if path is not None:
        user = yaml.safe_load(Path(path).read_text()) or {}
        cfg = _deep_merge(cfg, user)

    for key, value in overrides.items():
        if value is None:
            continue
        parts = key.split("__")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key in ("processed_dir", "features_dir", "artifacts_dir"):
        resolve(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

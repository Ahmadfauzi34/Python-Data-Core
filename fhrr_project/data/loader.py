"""
Loader dataset FHRR.

Pakai:
    from data.loader import load_dataset, list_datasets
    ds = load_dataset("default")          # by name (cari di datasets/)
    ds = load_dataset("/abs/path/x.yaml") # by path
"""
from __future__ import annotations
import os
from typing import Any

import yaml

from data.schema import assert_valid

_DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


def list_datasets() -> list[str]:
    """List nama dataset yang tersedia di datasets/ (tanpa .yaml)."""
    if not os.path.isdir(_DATASETS_DIR):
        return []
    out = []
    for f in sorted(os.listdir(_DATASETS_DIR)):
        if f.endswith((".yaml", ".yml")):
            out.append(os.path.splitext(f)[0])
    return out


def _resolve_path(name_or_path: str) -> str:
    if os.path.isabs(name_or_path) or os.sep in name_or_path:
        return name_or_path
    for ext in (".yaml", ".yml"):
        cand = os.path.join(_DATASETS_DIR, name_or_path + ext)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"Dataset {name_or_path!r} tidak ditemukan. Yang tersedia: {list_datasets()}"
    )


def load_dataset(name_or_path: str = "default", *, strict: bool = True) -> dict[str, Any]:
    """Load + validasi dataset YAML.

    Args:
        name_or_path: nama (mis. 'default') atau path absolut ke .yaml.
        strict: raise ValueError kalau ada error validasi.

    Returns dict siap-pakai untuk `runner.load_dataset(...)`.
    """
    path = _resolve_path(name_or_path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Konten {path} bukan mapping/dict YAML")

    # Default keys yang opsional jadi list kosong, supaya consumer tidak kena KeyError.
    for k in (
        "observations", "qa_pairs", "reasoning_patterns",
        "comprehension_tasks", "logical_pairs", "teaching_episodes",
        "explanation_templates",
    ):
        data.setdefault(k, [])
    data.setdefault("vocab", {}).setdefault("categories", {})
    data["vocab"].setdefault("poles", {})

    assert_valid(data, strict=strict)
    return data

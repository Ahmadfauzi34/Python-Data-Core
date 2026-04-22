"""
Loader dataset FHRR.

Pakai:
    from fhrr_project.data.loader import load_dataset, list_datasets
    ds = load_dataset("default")          # by name (cari di datasets/)
    ds = load_dataset("/abs/path/x.yaml") # by path
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

import yaml

from fhrr_project.data.schema import assert_valid

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
    # Ensure base directory is absolute
    base_dir = Path(_DATASETS_DIR).resolve()

    # Try matching by name first
    for ext in (".yaml", ".yml"):
        cand = base_dir / (name_or_path + ext)
        try:
            if cand.resolve().is_file() and cand.resolve().is_relative_to(base_dir):
                return str(cand.resolve())
        except (OSError, RuntimeError, ValueError):
            continue

    # Try treating as a direct path
    cand = Path(name_or_path)
    if not cand.is_absolute():
        cand = base_dir / cand

    try:
        resolved = cand.resolve()
        if resolved.is_file() and resolved.is_relative_to(base_dir):
            return str(resolved)
    except (OSError, RuntimeError, ValueError):
        pass

    raise FileNotFoundError(
        f"Dataset {name_or_path!r} tidak ditemukan atau akses dilarang. Yang tersedia: {list_datasets()}"
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

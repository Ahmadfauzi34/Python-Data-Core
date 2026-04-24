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
    """List nama dataset (file atau folder) yang tersedia di datasets/ (tanpa .yaml)."""
    if not os.path.isdir(_DATASETS_DIR):
        return []
    out = []
    for f in sorted(os.listdir(_DATASETS_DIR)):
        full_path = os.path.join(_DATASETS_DIR, f)
        if os.path.isdir(full_path):
            out.append(f)
        elif f.endswith((".yaml", ".yml")):
            out.append(os.path.splitext(f)[0])
    return sorted(list(set(out)))


def _resolve_path(name_or_path: str) -> str:
    # Ensure base directory is absolute
    base_dir = Path(_DATASETS_DIR).resolve()

    # Try matching by name as a folder first
    cand_dir = base_dir / name_or_path
    try:
        if cand_dir.resolve().is_dir() and cand_dir.resolve().is_relative_to(base_dir):
            return str(cand_dir.resolve())
    except (OSError, RuntimeError, ValueError):
        pass

    # Try matching by name as a file
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
        if (resolved.is_file() or resolved.is_dir()) and resolved.is_relative_to(base_dir):
            return str(resolved)
    except (OSError, RuntimeError, ValueError):
        pass

    raise FileNotFoundError(
        f"Dataset {name_or_path!r} tidak ditemukan atau akses dilarang. Yang tersedia: {list_datasets()}"
    )

def _deep_merge(target: dict, source: dict):
    """Recursively merges dictionary `source` into `target`."""
    for key, value in source.items():
        if isinstance(value, dict):
            node = target.setdefault(key, {})
            _deep_merge(node, value)
        elif isinstance(value, list):
            target.setdefault(key, []).extend(value)
        else:
            target[key] = value
    return target


def load_dataset(name_or_path: str = "default", *, strict: bool = True) -> dict[str, Any]:
    """Load + validasi dataset YAML, mendukung mode folder (Multi-File).

    Args:
        name_or_path: nama (mis. 'default') atau path absolut ke .yaml atau folder.
        strict: raise ValueError kalau ada error validasi.

    Returns dict siap-pakai untuk `runner.load_dataset(...)`.
    """
    path = _resolve_path(name_or_path)

    merged_data = {}

    if os.path.isdir(path):
        # Multi-file loading
        files_to_load = []
        for root, _, files in os.walk(path):
            for f in sorted(files):
                if f.endswith((".yaml", ".yml")):
                    files_to_load.append(os.path.join(root, f))

        if not files_to_load:
            raise ValueError(f"Folder {path} tidak memiliki file YAML")

        for filepath in files_to_load:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data is not None:
                if not isinstance(data, dict):
                    raise ValueError(f"Konten {filepath} bukan mapping/dict YAML")
                _deep_merge(merged_data, data)
    else:
        # Single-file loading
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is not None:
            if not isinstance(data, dict):
                raise ValueError(f"Konten {path} bukan mapping/dict YAML")
            merged_data = data

    data = merged_data

    if data.get("vocab") is None:
        data["vocab"] = {}

    # Default keys yang opsional jadi list kosong, supaya consumer tidak kena KeyError.
    for k in (
        "observations", "qa_pairs", "reasoning_patterns",
        "comprehension_tasks", "logical_pairs", "teaching_episodes",
        "explanation_templates",
    ):
        data.setdefault(k, [])
    if data.get("vocab") is None:
        data["vocab"] = {}
    data["vocab"].setdefault("categories", {})
    data["vocab"].setdefault("poles", {})

    assert_valid(data, strict=strict)
    return data

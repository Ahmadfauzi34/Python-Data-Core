# FHRR Chat

## Overview
Streamlit chat UI di atas sistem FHRR (Fractional Holographic Reduced Representation) yang memakai HRR + Knowledge Graph + topology untuk menjawab pertanyaan natural language.

## Struktur
- `main.py` — Streamlit chat UI (entry point)
- `fhrr_core.py` — fasad/re-export semua komponen FHRR (engine, runner, KG, loader, ingest, dst.)
- `fhrr_project/` — paket inti (semua sub-folder adalah paket Python dengan `__init__.py`)
  - `core/` — `engine.py`, `runner.py`, `topology.py`, `roles.py` (konstanta `Role.*`, `TripleKey.*`, `QUESTION_TO_ROLE`)
  - `data/`
    - `datasets/*.yaml` — sumber kebenaran dataset (default: `default.yaml`)
    - `schema.py` — `validate_dataset()` (cek struktur, ID duplikat, dangling reference, role tak dikenal)
    - `loader.py` — `load_dataset(name|path)` + `list_datasets()`
    - `ingest.py` — `ingest_dataset_to_kg(kg, ds)` (observation → triple KG)
    - `dataset.py` — shim back-compat (load YAML default)
  - `interface/`
    - `query_api.py` — `FHRRQueryInterface`
    - `text_normalizer.py` — normalisasi teks Indonesia: lowercase, strip `-kah/-lah/-pun`, varian verba `me-/di-/ter-/ber-` + nasal mutation, match by word-boundary
  - `memory/` — knowledge graph + open vocabulary
  - `agents/` — discoverer & self-improvement

## Menambah dataset baru
1. Buat `fhrr_project/data/datasets/<nama>.yaml` (struktur sama dengan `default.yaml`: `vocab`, `observations`, `qa_pairs`, …).
2. Pilih dari dropdown sidebar Streamlit, atau panggil `load_dataset("<nama>")`.
3. Validator akan menolak boot kalau ada error (ID duplikat, dangling reference, dll.); warning open-vocab tetap diizinkan.

## Run
Workflow `Start application`: `streamlit run main.py --server.port 5000 ...`

## Test
`python -m unittest discover tests -v` — smoke test (dataset, validator, ingest, end-to-end query) + unit test parser/normalizer.

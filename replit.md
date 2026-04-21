# FHRR Chat

## Overview
Streamlit chat UI di atas sistem FHRR (Fractional Holographic Reduced Representation) yang memakai HRR + Knowledge Graph + topology untuk menjawab pertanyaan natural language.

## Struktur
- `main.py` — Streamlit chat UI (entry point)
- `fhrr_core.py` — fasad/re-export semua komponen FHRR + helper `ingest_dataset_to_kg`
- `fhrr_project/` — paket inti
  - `core/` — `engine.py`, `runner.py`, `topology.py`
  - `data/dataset.py` — dataset penelitian
  - `interface/query_api.py` — `FHRRQueryInterface`
  - `memory/` — knowledge graph + open vocabulary
  - `agents/` — discoverer & self-improvement

## Run
Workflow `Start application`: `streamlit run main.py --server.port 5000 ...`

"""
Ingestion helpers — konversi dataset jadi triple KG.

Dipisah dari fhrr_core supaya logika "observation -> triple" hidup
di layer data, bukan di fasad re-export.
"""
from __future__ import annotations
from typing import Any

from memory.knowledge_graph import KnowledgeGraphIngestor, KGTriple


def ingest_dataset_to_kg(kg: KnowledgeGraphIngestor, dataset: dict[str, Any]) -> int:
    """Konversi observation jadi triple (agen, predikat, pasien) +
    relasi tambahan (di lokasi, pada waktu). Return jumlah triple ingested."""
    count = 0
    for obs in dataset.get("observations", []):
        b = obs.get("bindings", {})
        agen = b.get("agen") or b.get("subject")
        pred = b.get("predikat") or b.get("predicate")
        pas = b.get("pasien") or b.get("object")
        meta_base = {"obs_id": obs.get("id")}

        if agen and pred and pas:
            extra = {k: v for k, v in b.items() if k not in ("agen", "predikat", "pasien", "subject", "predicate", "object")}
            kg.ingest_triple(KGTriple(subject=agen, predicate=pred, object=pas, metadata={**meta_base, **extra}))
            count += 1
        if agen and b.get("lokasi"):
            kg.ingest_triple(KGTriple(subject=agen, predicate="di", object=b["lokasi"], metadata=dict(meta_base)))
            count += 1
        if agen and b.get("waktu"):
            kg.ingest_triple(KGTriple(subject=agen, predicate="pada", object=b["waktu"], metadata=dict(meta_base)))
            count += 1
    return count

"""
Ingestion helpers — konversi dataset jadi triple KG.

Dipisah dari fhrr_core supaya logika "observation -> triple" hidup
di layer data, bukan di fasad re-export.
"""
from __future__ import annotations
from typing import Any

from fhrr_project.core.roles import Role, get_binding
from fhrr_project.memory.knowledge_graph import KnowledgeGraphIngestor, KGTriple

# Role yang konsumsi-nya khusus (jadi predicate sendiri, bukan masuk metadata extra).
_CORE_ROLES = {
    Role.AGEN, Role.PREDIKAT, Role.PASIEN,
    Role.SUBJECT, Role.PREDICATE, Role.OBJECT,
    Role.LOKASI, Role.WAKTU,
}


def ingest_dataset_to_kg(kg: KnowledgeGraphIngestor, dataset: dict[str, Any]) -> int:
    """Konversi observation jadi triple (agen, predikat, pasien) +
    relasi tambahan (di lokasi, pada waktu). Return jumlah triple ingested."""
    count = 0
    for obs in dataset.get("observations", []):
        b = obs.get("bindings", {})
        agen = get_binding(b, Role.AGEN)
        pred = get_binding(b, Role.PREDIKAT)
        pas = get_binding(b, Role.PASIEN)
        meta_base = {"obs_id": obs.get("id")}

        if agen and pred and pas:
            extra = {k: v for k, v in b.items() if k not in _CORE_ROLES}
            kg.ingest_triple(KGTriple(subject=agen, predicate=pred, object=pas, metadata={**meta_base, **extra}))
            count += 1
        if agen and b.get(Role.LOKASI):
            kg.ingest_triple(KGTriple(subject=agen, predicate="di", object=b[Role.LOKASI], metadata=dict(meta_base)))
            count += 1
        if agen and b.get(Role.WAKTU):
            kg.ingest_triple(KGTriple(subject=agen, predicate="pada", object=b[Role.WAKTU], metadata=dict(meta_base)))
            count += 1
    return count

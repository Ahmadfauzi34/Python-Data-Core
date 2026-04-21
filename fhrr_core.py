"""
fhrr_core — fasad/re-export semua komponen FHRR dari paket fhrr_project/.

Tujuannya supaya kode di root (main.py) bisa pakai gaya:
    from fhrr_core import FHRRResearchRunner, fhrr_research_dataset, ...
tanpa peduli struktur folder internal.
"""
import os
import sys

# Tambahkan fhrr_project/ ke sys.path supaya import internalnya
# (mis. `from core.engine import ...`) tetap jalan.
_PKG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fhrr_project")
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from core.engine import FHRREngine  # noqa: E402
from core.runner import (  # noqa: E402
    FHRRResearchRunner,
    FHRRResearchTrainer,
    FHRREvaluator,
    TrainingResult,
)
from core.topology import FHRRTopologicalLayer  # noqa: E402
from data.dataset import fhrr_research_dataset  # noqa: E402
from interface.query_api import FHRRQueryInterface, QueryResult  # noqa: E402
from memory.knowledge_graph import KnowledgeGraphIngestor, KGTriple  # noqa: E402
from memory.open_vocab import (  # noqa: E402
    OpenVocabularyExtension,
    extend_engine_open_vocab,
)
from agents.discoverer import SelfSupervisedDiscovery  # noqa: E402
from agents.improver import SelfImprovementEngine, AssessmentReport  # noqa: E402
from data.loader import load_dataset, list_datasets  # noqa: E402
from data.ingest import ingest_dataset_to_kg  # noqa: E402
from data.schema import validate_dataset, assert_valid  # noqa: E402


__all__ = [
    "FHRREngine",
    "FHRRResearchRunner",
    "FHRRResearchTrainer",
    "FHRREvaluator",
    "TrainingResult",
    "FHRRTopologicalLayer",
    "fhrr_research_dataset",
    "FHRRQueryInterface",
    "QueryResult",
    "KnowledgeGraphIngestor",
    "KGTriple",
    "OpenVocabularyExtension",
    "extend_engine_open_vocab",
    "SelfSupervisedDiscovery",
    "SelfImprovementEngine",
    "AssessmentReport",
    "ingest_dataset_to_kg",
    "load_dataset",
    "list_datasets",
    "validate_dataset",
    "assert_valid",
]

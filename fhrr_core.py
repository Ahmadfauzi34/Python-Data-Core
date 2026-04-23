"""
fhrr_core — fasad/re-export semua komponen FHRR dari paket fhrr_project/.

Tujuannya supaya kode di root (main.py) bisa pakai gaya:
    from fhrr_core import FHRRResearchRunner, fhrr_research_dataset, ...
tanpa peduli struktur folder internal.
"""

from fhrr_project.core.engine import FHRREngine  # noqa: E402
from fhrr_project.core.runner import (  # noqa: E402
    FHRRResearchRunner,
    FHRRResearchTrainer,
    FHRREvaluator,
    TrainingResult,
)
from fhrr_project.core.topology import FHRRTopologicalLayer  # noqa: E402
from fhrr_project.data.dataset import fhrr_research_dataset  # noqa: E402
from fhrr_project.interface.query_api import FHRRQueryInterface, QueryResult  # noqa: E402
from fhrr_project.memory.knowledge_graph import KnowledgeGraphIngestor, KGTriple  # noqa: E402
from fhrr_project.memory.open_vocab import (  # noqa: E402
    OpenVocabularyExtension,
    extend_engine_open_vocab,
)
from fhrr_project.agents.discoverer import SelfSupervisedDiscovery  # noqa: E402
from fhrr_project.agents.improver import SelfImprovementEngine, AssessmentReport  # noqa: E402
from fhrr_project.data.loader import load_dataset, list_datasets  # noqa: E402
from fhrr_project.data.ingest import ingest_dataset_to_kg  # noqa: E402
from fhrr_project.data.schema import validate_dataset, assert_valid  # noqa: E402


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

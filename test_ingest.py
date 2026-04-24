import sys
from fhrr_project.data.loader import load_dataset
from fhrr_project.data.ingest import ingest_dataset_to_kg
from fhrr_project.core.runner import FHRRResearchRunner
from fhrr_project.memory.open_vocab import extend_engine_open_vocab
from fhrr_project.memory.knowledge_graph import KnowledgeGraphIngestor

dataset = load_dataset('default')
runner = FHRRResearchRunner(dim=4096)
runner.load_dataset(dataset)

open_vocab = extend_engine_open_vocab(runner.engine)
kg = KnowledgeGraphIngestor(runner.engine, open_vocab)
n_triples = ingest_dataset_to_kg(kg, dataset)
runner.attach_kg(kg)

text = "Budi makan apel di rumah. Ani membaca buku di sekolah."
print("Ingesting text:", text)
new_memories = runner.ingest_unstructured_text(text)
print("New memories extracted:", new_memories)

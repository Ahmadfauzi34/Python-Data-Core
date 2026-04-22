import time
import numpy as np
from fhrr_project.core.engine import FHRREngine
from fhrr_project.core.topology import FHRRTopologicalLayer
from fhrr_project.agents.discoverer import SelfSupervisedDiscovery
from fhrr_project.agents.improver import SelfImprovementEngine
from fhrr_project.memory.knowledge_graph import KnowledgeGraphIngestor
from fhrr_project.memory.open_vocab import extend_engine_open_vocab

def run_benchmark():
    dim = 1024
    engine = FHRREngine(dim=dim)
    n_categories = 100
    tokens_per_cat = 10

    print(f"Populating engine with {n_categories} categories and {tokens_per_cat} tokens each...")
    for i in range(n_categories):
        cat_name = f"cat_{i}"
        proto = engine.alloc()
        for j in range(tokens_per_cat):
            engine.add_token(f"token_{i}_{j}", cat_name, prototype=proto)

    topo = FHRRTopologicalLayer(engine)
    # Force build stalks
    topo.sheaf.build_stalks()
    # We want many sheaf_isolation gaps, so we don't build base space or we build it with very high threshold
    # topo.sheaf.build_base_space(cooccurrence_threshold=1.0) # all will be isolated

    # Actually _detect_gaps needs base_adj to be truthy to even consider sheaf_isolation
    topo.sheaf.base_adj['dummy'] = set()

    open_vocab = extend_engine_open_vocab(engine)
    kg = KnowledgeGraphIngestor(engine, open_vocab)
    discoverer = SelfSupervisedDiscovery(engine)
    improver = SelfImprovementEngine(engine, topo, discoverer, kg)

    gaps = improver._detect_gaps()
    sheaf_gaps = [g for g in gaps if g['type'] == 'sheaf_isolation']
    print(f"Detected {len(sheaf_gaps)} sheaf_isolation gaps.")

    # Measure _generate_suggestions
    # We might want to remove the [:5] cap for benchmarking purposes or just see how it scales with N
    start_time = time.time()
    for _ in range(100):
        improver._generate_suggestions(gaps)
    end_time = time.time()

    print(f"Time taken for 100 calls to _generate_suggestions: {end_time - start_time:.4f}s")

if __name__ == "__main__":
    run_benchmark()

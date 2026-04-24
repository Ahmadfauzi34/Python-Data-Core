import time
import numpy as np
from fhrr_project.core.engine import FHRREngine
from fhrr_project.memory.consolidation import MetaCognitiveConsolidator

engine = FHRREngine(dim=4096)
engine.add_role("predikat")
engine.add_role("pasien")
engine.add_role("atribut")

for i in range(100):
    engine.add_token(f"t{i}", "cat")

consolidator = MetaCognitiveConsolidator(engine, "")

pool = []
for i in range(500):
    pool.append({
        'from': f"t{np.random.randint(0, 10)}",
        'to': f"t{np.random.randint(10, 20)}",
        'vector': np.random.randn(4096)
    })

t0 = time.time()
consolidator._cluster_pool(pool, 0.35, "test")
t1 = time.time()

print(f"_cluster_pool took {t1 - t0:.4f} seconds")

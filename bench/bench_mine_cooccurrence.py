import time
import numpy as np
from fhrr_project.core.engine import FHRREngine
from fhrr_project.agents.discoverer import SelfSupervisedDiscovery

engine = FHRREngine(dim=4096)
for i in range(100):
    engine.add_token(f"t{i}", "cat")

discoverer = SelfSupervisedDiscovery(engine)
discoverer.ingest_corpus([" ".join([f"t{np.random.randint(0, 100)}" for _ in range(20)]) for _ in range(500)])

t0 = time.time()
discoverer.mine_cooccurrence()
t1 = time.time()

print(f"mine_cooccurrence took {t1 - t0:.4f} seconds")

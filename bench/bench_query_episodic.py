import time
import numpy as np
from fhrr_project.core.engine import FHRREngine

engine = FHRREngine(dim=4096)
engine.add_role("agen")
engine.add_role("predikat")
engine.add_role("pasien")

for i in range(100):
    engine.add_token(f"t{i}", "cat")

for i in range(5000):
    engine.episodic_buffer.append({
        'vector': np.random.randn(4096),
        'decay_factor': 0.99,
        'access_count': 0
    })

engine._episodic_vectors = np.array([e['vector'] for e in engine.episodic_buffer])
engine._episodic_C_v = np.cos(engine._episodic_vectors)
engine._episodic_S_v = np.sin(engine._episodic_vectors)
engine._episodic_decay = np.array([e['decay_factor'] for e in engine.episodic_buffer])

q_vec = np.random.randn(4096)

t0 = time.time()
engine.query_episodic(q_vec)
t1 = time.time()
print(f"query_episodic without maintained buffer took {t1 - t0:.4f} seconds for 5000 entries")

t0 = time.time()
C_q = np.cos(q_vec)
S_q = np.sin(q_vec)
sim_arr = (engine._episodic_C_v @ C_q + engine._episodic_S_v @ S_q) / 4096
compensated = sim_arr * engine._episodic_decay
best_idx = np.argmax(compensated)
t1 = time.time()
print(f"query_episodic with maintained buffer took {t1 - t0:.4f} seconds for 5000 entries")

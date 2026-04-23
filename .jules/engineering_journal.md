**Section 2: Architectural Decisions (ADR)**

## 2024-05-18 - [⬡ Carbo] - [Modular Dataset Architecture]
**Context:** The user requested to split the `default.yaml` dataset to make it easier to maintain.
**Decision:** I split the monolithic `default.yaml` into several specialized files (`vocab.yaml`, `observations.yaml`, `reasoning_patterns.yaml`, etc.) and placed them into a `fhrr_project/data/datasets/default/` directory.
**Consequences:** The dataset is now modular, improving maintainability and adhering to the multi-file loading capabilities already built into `loader.py`.

**Section 3: Future Ideas**

## 2024-05-18 - [⬡ Carbo] - [The Simulation Space (Mental Sandbox)]
**Context:** The user requested an architectural concept for a "Simulation Space", allowing the AI to simulate roleplay scenarios before outputting them, essentially creating an epistemic fork of the mental state.
**Decision:** Designed and built `SimulationSpace` in `fhrr_project/agents/simulation.py`. It leverages the FHRREngine to initialize a base state and then projects hypothetical actions via complex phase bundling (`c_state + c_action`). Each scenario is then evaluated using a dual-metric system: Goal Proximity (Cosine similarity to a target epistemic state) and Topological Coherence (checking the resulting vector against the Sheaf's global section consistency). Finally, it implements "wavefunction collapse" to return the optimal path.
**Consequences:** We now have a zero-copy prediction engine that allows agents to evaluate the future logical consistency of their actions without mutating the core memory/knowledge graph. The math is elegant (complex plane superposition) and adheres strictly to DRY and single-responsibility principles.

## 2024-05-18 - [⬡ Carbo] - [Spectral Geometry Affinity Vectorization]
**Context:** The `build_affinity` method in `fhrr_project/core/topology.py` used a nested double Python loop (`O(N^2)`) to compute the element-wise exponential weighting for the K-Nearest Neighbors topology adjacency matrix. This caused ~7 second delays for just 500 tokens.
**Decision:** Rewrote the entire method into pure linear algebra. Generated the full NxN pairwise similarity matrix instantaneously using `(C @ C.T + S @ S.T) / dim`. Applied the KNN threshold filter via `np.partition(..., axis=1)`, and applied the affinity exponential scaling directly on the boolean mask.
**Consequences:** Complete elimination of the Python loops, bringing the operation down to ~0.16s (>40x speedup), making large-scale topology clustering operations vastly more fluid.

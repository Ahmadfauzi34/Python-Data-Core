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

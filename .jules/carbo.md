**Section 1: Architectural Decisions (ADR)**

## 2024-05-18 - [⬡ Carbo] - [Modular Dataset Architecture]
**Context:** The user requested to split the `default.yaml` dataset to make it easier to maintain.
**Decision:** I split the monolithic `default.yaml` into several specialized files (`vocab.yaml`, `observations.yaml`, `reasoning_patterns.yaml`, etc.) and placed them into a `fhrr_project/data/datasets/default/` directory.
**Consequences:** The dataset is now modular, improving maintainability and adhering to the multi-file loading capabilities already built into `loader.py`.

**Section 2: The Idea Forge**


## 2024-05-18 - [⬡ Carbo] - [The Simulation Space (Mental Sandbox)]
**Vision:** An epistemic fork mechanism. Instead of reactive output, the agent explores a "Mental Sandbox" by projecting multiple semantic trajectories and evaluating their topological stability.
**Architecture:** `SimulationSpace` (in `agents/simulation.py`).
1. **Projection:** `project_action` binds potential actions into a temporary state using FHRR superposition.
2. **Evaluation:** `evaluate_scenarios` calculates Epistemic Reward (vector similarity to goal) and Topological Coherence (sheaf constraints).
3. **Collapse:** Returns the highest-scoring reality.
**Blockers:** Currently requires explicit goal vectors. In the future, this could be combined with a self-supervised policy gradient.

## 2024-05-18 - [⬡ Carbo] - [Spectral Geometry Affinity Vectorization]
**Context:** The `build_affinity` method in `fhrr_project/core/topology.py` used a nested double Python loop (`O(N^2)`) to compute the element-wise exponential weighting for the K-Nearest Neighbors topology adjacency matrix. This caused ~7 second delays for just 500 tokens.
**Decision:** Rewrote the entire method into pure linear algebra. Generated the full NxN pairwise similarity matrix instantaneously using `(C @ C.T + S @ S.T) / dim`. Applied the KNN threshold filter via `np.partition(..., axis=1)`, and applied the affinity exponential scaling directly on the boolean mask.
**Consequences:** Complete elimination of the Python loops, bringing the operation down to ~0.16s (>40x speedup), making large-scale topology clustering operations vastly more fluid.

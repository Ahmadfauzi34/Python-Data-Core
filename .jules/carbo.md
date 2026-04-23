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

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

## 2024-05-18 - [⬡ Carbo] - [Rule-based Semantic NLP Extraction Blueprint]
**Vision:** Text ingestion should be intelligent enough to bridge unstructured text to FHRR bindings without massive NLP overhead, using prepositional cues to assign semantic roles and categorize Out-Of-Vocabulary words.
**Architecture:**
1. **Extraction:** A state-machine in `_extract_entities_and_roles` routes tokens based on prepositional triggers (`di` -> Lokasi, `oleh` -> Agen).
2. **OOTV Handling:** In `ingest_document`, if a token is unknown, we map its extracted semantic role to a lexical category guess (Predikat -> `aksi`, Lokasi -> `tempat`).
3. **Storage:** Ingests into both the Episodic Buffer (short-term phase state) and Knowledge Graph (long-term explicit graph).
**Blockers:** Current heuristics only support S-P-O and simple prepositional modifiers. Deep recursive structures (anak yang memakai baju merah memakan apel) still require a proper dependency parser like Spacy in the future.

## 2024-05-18 - [⬡ Carbo] - [Discoverer Clustering Vectorization]
**Context:** The `induce_roles` method in `SelfSupervisedDiscovery` utilized a primitive 1D K-Means loop. It iterated 15 times, and within each iteration, it iterated over N tokens and K centroids, redundantly using `np.pad`, `np.dot`, and scalar norms.
**Decision:** Applied the same linear algebra vectorization principles used in topology. Built a padded, normalized `(N, max_len)` feature matrix `X` upfront. Substituted the nested loops with `sims = X @ centroids_norm.T` to calculate Euclidean cosine similarities simultaneously, and updated centroids using NumPy's highly efficient boolean masking `X[labels == k].mean(axis=0)`.
**Consequences:** Replaced hundreds of scalar math iterations and Python list comprehensions with three robust, highly optimized BLAS matrix routines, maintaining identical unsupervised accuracy.

## 2024-05-18 - [⬡ Carbo] - [Refactor Absolute Imports in fhrr_core.py]
**Context:** The linter reported "Cannot resolve imported module" errors in `fhrr_core.py` because the facade pattern used a dynamic `sys.path.insert(0, _PKG_DIR)` to import components. This violates static analysis resolution patterns and makes navigation harder.
**Decision:** Removed the `sys.path` hack. Refactored all internal component imports to use fully qualified absolute package imports compliant with PEP-8 (e.g., `from fhrr_project.core.engine import FHRREngine`).
**Consequences:** Static analysis tools and linters (like Pyright/Mypy) can now successfully resolve all modules in `fhrr_core.py`. The architecture follows stricter packaging norms without functional regressions.

## 2024-05-18 - [⬡ Carbo] - [Dataset Enrichment (Breadth & Depth)]
**Context:** The modular `default` dataset contained very primitive lexical domains (mostly just animals and simple actions). To improve the agent's semantic comprehension and multi-hop reasoning, the dataset needs greater breadth and depth.
**Decision:** Enriched all modules across the board. Added new categories (`profesi`, `kendaraan`, `cuaca`) and extensive vocabularies to `vocab.yaml`. Created corresponding episodic memories in `observations.yaml`. Added complex multi-hop transitive and causal rules to `reasoning_patterns.yaml` (e.g. spatial transitives: lab -> school). Updated `comprehension_tasks.yaml` to rigorously test these new logical connections.
**Consequences:** The FHRR engine now has a significantly denser semantic topology and Knowledge Graph, enabling it to handle more complex real-world logic patterns and out-of-the-box generalizations.

## 2024-05-18 - [⬡ Carbo] - [Fix Linter Typings in text_ingestor.py]
**Context:** The `text_ingestor.py` file attempted to use defensive programming for an internal module import (`KnowledgeGraphIngestor`), falling back to assigning classes to `None` if the import failed. This caused strict linters/type checkers to complain about incompatible type assignments (`None` assigned to class).
**Decision:** Because `KnowledgeGraphIngestor` is a core internal component guaranteed to exist alongside `text_ingestor.py`, the `try-except` block is an anti-pattern. Removed it and replaced it with a direct, explicit import.
**Consequences:** Cleans up the code, removes false-positive linter errors, and enforces tighter module coupling where appropriate.

## 2024-05-18 - [⬡ Carbo] - [Meta-Cognitive Consolidator (Sleep Phase)]
**Vision:** Holographic Memory Consolidation. The AI should not just apply logic, it should discover logic by extracting repeated phase transformations across varying semantic contexts.
**Architecture:** `MetaCognitiveConsolidator` in `fhrr_project/memory/consolidation.py`.
1. **Extraction:** Retrieves temporary transformations experienced by the agent.
2. **Clustering:** Uses vectorized linear algebra to cluster similar transformation vectors ($T \approx A \oslash B$).
3. **Induction:** Generates a semantic law if a cluster has enough evidence.
4. **Self-Modification:** Persists the rule into `fhrr_project/data/datasets/default/reasoning_patterns.yaml` permanently.
**Blockers:** Current extraction relies on explicitly logged transforms. Future iterations should dynamically unbind chronological episodic events (e.g., $Event_2 \oslash Event_1$) to find hidden temporal causations.

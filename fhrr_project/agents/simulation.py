import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from fhrr_project.core.engine import FHRREngine
from fhrr_project.core.topology import FHRRTopologicalLayer

@dataclass
class SimulationScenario:
    id: str
    action_bindings: Dict[str, str]
    action_vec: np.ndarray
    resulting_state: np.ndarray
    coherence_score: float = 0.0
    reward_score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

class SimulationSpace:
    """
    Mental Sandbox: A zero-copy (or lightweight) simulation layer where the agent
    can project possible future actions (roleplay) and evaluate their topological
    and epistemic consequences before collapsing the wavefunction into reality.
    """
    def __init__(self, main_engine: FHRREngine, topology_layer: Optional[FHRRTopologicalLayer] = None, reward_weight: float = 0.6, coherence_weight: float = 0.4):
        self.main_engine = main_engine
        self.topology = topology_layer
        self.scenarios: List[SimulationScenario] = []
        self._base_state: Optional[np.ndarray] = None
        self._goal_state: Optional[np.ndarray] = None
        self._base_bindings: Dict[str, str] = {}
        self.reward_weight = reward_weight
        self.coherence_weight = coherence_weight

    def initialize_state(self, current_state_bindings: Dict[str, str], goal_bindings: Optional[Dict[str, str]] = None):
        """Snapshots the current context to begin a simulation fork."""
        self._base_bindings = current_state_bindings.copy()
        self._base_state = self.main_engine.encode(current_state_bindings)
        if self._base_state is None:
            raise ValueError("Failed to encode current state bindings")

        if goal_bindings:
            self._goal_state = self.main_engine.encode(goal_bindings)
        else:
            self._goal_state = None

        self.scenarios.clear()

    def project_action(self, action_id: str, action_bindings: Dict[str, str], metadata: Optional[Dict] = None):
        """Forks a new branch in the simulation space by applying an action to the base state."""
        if self._base_state is None:
            raise RuntimeError("Simulation space not initialized. Call initialize_state first.")

        action_vec = self.main_engine.encode(action_bindings)
        if action_vec is None:
            return None

        # Bind the action to the current state (Simulation step)
        # FHRR Bundling/Binding logic depending on how we represent sequential actions.
        # Here we use bundle (superposition) to represent the "world state + action"
        c_state = np.exp(1j * self._base_state)
        c_action = np.exp(1j * action_vec)
        c_result = c_state + c_action

        # Normalize back to phase
        resulting_state = np.angle(c_result)

        scenario = SimulationScenario(
            id=action_id,
            action_bindings=action_bindings,
            action_vec=action_vec,
            resulting_state=resulting_state,
            metadata=metadata or {}
        )
        self.scenarios.append(scenario)
        return scenario

    def evaluate_scenarios(self) -> List[SimulationScenario]:
        """Scores all projected scenarios based on goal proximity and topological coherence."""
        if not self.scenarios:
            return []

        for scenario in self.scenarios:
            # 1. Goal Proximity (Epistemic Reward)
            if self._goal_state is not None:
                # Cosine similarity between resulting state and goal state
                sim = self.main_engine.sim(scenario.resulting_state, self._goal_state)
                scenario.reward_score = sim
            else:
                scenario.reward_score = 0.5 # Neutral if no goal

            # 2. Topological Coherence (Logical consistency)
            if self.topology and len(self.topology.sheaf.stalks) > 0:
                # We decode the resulting state to see if the proposed roleplay violates sheaf constraints
                decoded = self.main_engine.decode(scenario.resulting_state, threshold=0.3)

                # Check consistency
                assignment = {}
                for role, (filler, conf) in decoded.items():
                    tok_idx = self.main_engine.get_token_idx(filler)
                    if tok_idx is not None:
                        cat = self.main_engine.token_categories[tok_idx]
                        assignment[cat] = self.main_engine.token_phases[tok_idx]

                if assignment:
                    is_consistent, violations = self.topology.sheaf.global_section_consistency(assignment, tol=0.4)
                    # Normalize by the number of categories actually assigned, not the entire base_adj
                    num_constraints = len(assignment)
                    coherence = 1.0 - (len(violations) / max(1, num_constraints))
                    scenario.coherence_score = max(0.0, coherence)
                else:
                    scenario.coherence_score = 0.5
            else:
                scenario.coherence_score = 1.0 # Perfect coherence if no topology checks

        # Sort scenarios by a combined metric
        self.scenarios.sort(key=lambda x: (x.reward_score * self.reward_weight) + (x.coherence_score * self.coherence_weight), reverse=True)
        return self.scenarios

    def collapse(self) -> Optional[SimulationScenario]:
        """Wavefunction collapse: selects the highest scoring scenario to manifest in reality."""
        evaluated = self.evaluate_scenarios()
        if not evaluated:
            return None
        return evaluated[0]

    def commit(self, scenario: SimulationScenario, kg_ingestor: Optional[Any] = None) -> bool:
        """
        Manifesasikan scenario yang dipilih ke dalam memori mesin utama.
        Ini mengubah state FHRREngine dan KnowledgeGraph.
        """
        if scenario is None:
            return False

        # Store to episodic memory
        self.main_engine.store_episodic(scenario.resulting_state, metadata=scenario.metadata)

        # If KG ingestor is provided, commit the action if it resembles a valid triple
        if kg_ingestor and 'aksi' in scenario.action_bindings and 'target' in scenario.action_bindings:
            # We assume agent identity is preserved from base state (simplification)
            try:
                from fhrr_project.memory.knowledge_graph import KGTriple
                # Try to extract the agent from the base state, fallback to a generic placeholder if missing
                agent_id = self._base_bindings.get('agen', 'unknown_agent')

                triple = KGTriple(
                    subject=agent_id,
                    predicate=scenario.action_bindings['aksi'],
                    object=scenario.action_bindings['target'],
                    metadata={"source": "simulation_commit"}
                )
                kg_ingestor.ingest_triple(triple)
            except Exception as e:
                print(f"[SimulationSpace] Failed to commit to KG: {e}")

        return True

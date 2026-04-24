import unittest
import numpy as np
from fhrr_project.core.engine import FHRREngine
from fhrr_project.agents.simulation import SimulationSpace

class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.engine = FHRREngine(dim=512)
        self.engine.add_role("aksi")
        self.engine.add_token("sapa", "aksi")
        self.engine.add_token("marah", "aksi")
        self.sim = SimulationSpace(self.engine, topology_layer=None)

    def test_simulation_workflow(self):
        self.sim.initialize_state({"agen": "user", "aksi": "sapa"}, goal_bindings={"aksi": "sapa"})

        s1 = self.sim.project_action("act1", {"aksi": "sapa"})
        s2 = self.sim.project_action("act2", {"aksi": "marah"})

        self.assertIsNotNone(s1)
        self.assertIsNotNone(s2)

        best = self.sim.collapse()
        self.assertIsNotNone(best)
        self.assertEqual(best.id, "act1") # Because it perfectly matches the goal state

        # Test commit
        ep_count_before = len(self.engine.episodic_buffer)

        # Create a mock KG Ingestor to verify agent ID mapping
        class MockKG:
            def __init__(self):
                self.triples = []
            def ingest_triple(self, triple):
                self.triples.append(triple)

        mock_kg = MockKG()

        # Test agent fallback and commit logic
        best.action_bindings['target'] = 'budi' # Ensure target exists for KG
        self.sim.commit(best, mock_kg)
        self.assertEqual(len(self.engine.episodic_buffer), ep_count_before + 1)
        self.assertEqual(len(mock_kg.triples), 1)
        # Check if the agent ID from initialize_state ("user") was correctly passed
        self.assertEqual(mock_kg.triples[0].subject, "user")

        # Test Coherence scoring behavior (mocking a violation)
        class MockSheaf:
            def __init__(self):
                self.stalks = {'mock': 'mock'}
                self.base_adj = {'mock': ['mock']}
            def global_section_consistency(self, assignment, tol):
                # Return false and 1 violation
                return False, ["violation1"]

        class MockTopology:
            def __init__(self):
                self.sheaf = MockSheaf()

        sim_with_topo = SimulationSpace(self.engine, MockTopology())
        sim_with_topo.initialize_state({"agen": "user", "aksi": "sapa"}, goal_bindings={"aksi": "sapa"})
        s3 = sim_with_topo.project_action("act_bad", {"aksi": "marah"})
        sim_with_topo.evaluate_scenarios()

        # With 1 violation and 1 assignment (aksi), coherence should be 1.0 - (1/1) = 0.0
        self.assertEqual(s3.coherence_score, 0.0)

if __name__ == '__main__':
    unittest.main()

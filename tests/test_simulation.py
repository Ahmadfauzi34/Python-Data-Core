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
        self.sim.initialize_state({"aksi": "sapa"}, goal_bindings={"aksi": "sapa"})

        s1 = self.sim.project_action("act1", {"aksi": "sapa"})
        s2 = self.sim.project_action("act2", {"aksi": "marah"})

        self.assertIsNotNone(s1)
        self.assertIsNotNone(s2)

        best = self.sim.collapse()
        self.assertIsNotNone(best)
        self.assertEqual(best.id, "act1") # Because it perfectly matches the goal state

        # Test commit
        ep_count_before = len(self.engine.episodic_buffer)
        self.sim.commit(best)
        self.assertEqual(len(self.engine.episodic_buffer), ep_count_before + 1)

if __name__ == '__main__':
    unittest.main()

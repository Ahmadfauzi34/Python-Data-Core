import unittest
import numpy as np
from fhrr_project.core.engine import FHRREngine
from fhrr_project.agents.improver import SelfImprovementEngine
from fhrr_project.core.topology import FHRRTopologicalLayer

class TestEngineOptimization(unittest.TestCase):
    def setUp(self):
        self.engine = FHRREngine(dim=1024)

    def test_sim_accuracy(self):
        a = self.engine.alloc()
        b = a.copy()
        self.assertAlmostEqual(self.engine.sim(a, b), 1.0, places=5)

        c = (a + np.pi) % (2*np.pi)
        self.assertAlmostEqual(self.engine.sim(a, c), -1.0, places=5)

    def test_improver_generate_suggestions_shape(self):
        topo = FHRRTopologicalLayer(self.engine)
        topo.sheaf.base_adj = {'dummy': set()}

        for i in range(5):
            self.engine.add_token(f"t_{i}", f"cat_{i}")
        topo.sheaf.build_stalks()

        class DummyDiscoverer:
            pass
        class DummyKG:
            pass

        improver = SelfImprovementEngine(self.engine, topo, DummyDiscoverer(), DummyKG())
        gaps = [{'type': 'sheaf_isolation', 'category': 'cat_0'}]
        suggestions = improver._generate_suggestions(gaps)
        self.assertTrue(len(suggestions) > 0)
        self.assertEqual(suggestions[0]['target_gap']['type'], 'sheaf_isolation')

    def test_vsa_operations(self):
        a = self.engine.alloc()
        b = self.engine.alloc()

        bound = self.engine.bind(a, b)
        unbound = self.engine.unbind(bound, a)
        self.assertTrue(self.engine.sim(unbound, b) > 0.99)

        c = self.engine.alloc()
        bundled = self.engine.bundle([a, b, c])
        self.assertTrue(self.engine.sim(bundled, a) > 0.3)
        self.assertTrue(self.engine.sim(bundled, b) > 0.3)
        self.assertTrue(self.engine.sim(bundled, c) > 0.3)

if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
import os
import sys

# Ensure fhrr_project is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, "fhrr_project") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "fhrr_project"))

from core.engine import FHRREngine

class TestLSHHash(unittest.TestCase):
    def setUp(self):
        self.dim = 128
        self.n_tables = 4
        self.n_bins = 10
        self.engine = FHRREngine(dim=self.dim, n_hash_tables=self.n_tables, n_bins=self.n_bins)

    def test_lsh_hash_determinism(self):
        """Verify that the same vector and table index always yield the same hash."""
        vec = np.random.uniform(-np.pi, np.pi, self.dim)
        h1 = self.engine._lsh_hash(vec, 0)
        h2 = self.engine._lsh_hash(vec, 0)
        self.assertEqual(h1, h2)

    def test_lsh_hash_range(self):
        """Verify that the output is always within [0, n_bins - 1]."""
        for _ in range(100):
            vec = np.random.uniform(-100, 100, self.dim)  # Use wide range of values
            for t in range(self.n_tables):
                h = self.engine._lsh_hash(vec, t)
                self.assertGreaterEqual(h, 0)
                self.assertLess(h, self.n_bins)

    def test_lsh_hash_logic(self):
        """Verify the binning logic with specific projections and vectors."""
        # Force a specific projection: all zeros except the first element is 1.0
        # proj = [1.0, 0.0, ...]
        self.engine.lsh_projections[0] = np.zeros(self.dim)
        self.engine.lsh_projections[0][0] = 1.0

        # Case 1: cos(vec) dot proj = 1.0
        # scalar = 1.0 -> bin_idx = int((1.0 + 1.0) / 2.0 * n_bins) = n_bins
        # Clamped to n_bins - 1
        vec1 = np.zeros(self.dim)  # cos(0) = 1.0
        h1 = self.engine._lsh_hash(vec1, 0)
        self.assertEqual(h1, self.n_bins - 1)

        # Case 2: cos(vec) dot proj = -1.0
        # scalar = -1.0 -> bin_idx = int((-1.0 + 1.0) / 2.0 * n_bins) = 0
        vec2 = np.zeros(self.dim)
        vec2[0] = np.pi  # cos(pi) = -1.0
        h2 = self.engine._lsh_hash(vec2, 0)
        self.assertEqual(h2, 0)

        # Case 3: cos(vec) dot proj = 0.0
        # scalar = 0.0 -> bin_idx = int((0.0 + 1.0) / 2.0 * n_bins) = n_bins // 2
        vec3 = np.zeros(self.dim)
        vec3[0] = np.pi / 2.0  # cos(pi/2) = 0.0
        h3 = self.engine._lsh_hash(vec3, 0)
        self.assertEqual(h3, self.n_bins // 2)

    def test_lsh_hash_clamping(self):
        """Verify that extreme scalar values are clamped correctly."""
        # Use a projection that could lead to out-of-bounds scalar if not handled
        self.engine.lsh_projections[0] = np.ones(self.dim) * 100.0
        vec = np.zeros(self.dim)
        # scalar = np.dot(ones, ones * 100) = dim * 100
        # Logic: h = max(0, min(int(...), n_bins-1))
        h = self.engine._lsh_hash(vec, 0)
        self.assertEqual(h, self.n_bins - 1)

        self.engine.lsh_projections[0] = np.ones(self.dim) * -100.0
        h = self.engine._lsh_hash(vec, 0)
        self.assertEqual(h, 0)

if __name__ == "__main__":
    unittest.main()

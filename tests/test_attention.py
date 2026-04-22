import unittest
import numpy as np
import time
from fhrr_project.core.attention import CrossSequenceMemory, AdaptiveFHRRAttention

class TestAttention(unittest.TestCase):
    def test_cross_sequence_memory_initialization(self):
        mem = CrossSequenceMemory(n_heads=4, head_dim=1024)
        self.assertEqual(mem.ltm_a.shape, (4096,))
        self.assertEqual(mem.strength.shape, (4,))
        self.assertEqual(mem.age.shape, (4,))

    def test_memory_injection(self):
        mem = CrossSequenceMemory(n_heads=2, head_dim=4)
        ext = np.ones(8, dtype=np.float64) * np.pi
        mem.inject_prior(ext, gamma=0.5, n_heads=2, head_dim=4)
        active = mem.read_active()
        # Initial was 0, external is pi, gamma=0.5 -> expected around 0.5 * pi
        self.assertTrue(np.allclose(active, np.pi / 2))

    def test_attention_forward_shape_and_speed(self):
        n_heads = 8
        total_dim = 4096
        max_seq = 64
        seq_len = 32

        attn = AdaptiveFHRRAttention(n_heads=n_heads, total_dim=total_dim, max_seq=max_seq)

        # Create random FHRR sequence [-pi, pi]
        x_seq = np.random.uniform(-np.pi, np.pi, max_seq * total_dim).astype(np.float64)

        start = time.time()
        out = attn.forward(x_seq, seq_len=seq_len)
        elapsed = time.time() - start

        self.assertEqual(out.shape, (seq_len * total_dim,))

        # It should be extremely fast due to vectorization, easily < 0.1s
        self.assertTrue(elapsed < 0.2, f"Forward pass took too long: {elapsed:.4f}s")

        # Run a few more times to test memory swapping and consolidation mechanics
        for _ in range(5):
            out2 = attn.forward(x_seq, seq_len=seq_len)
            self.assertEqual(out2.shape, (seq_len * total_dim,))

if __name__ == '__main__':
    unittest.main()

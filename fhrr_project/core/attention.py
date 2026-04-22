import numpy as np
from numpy.typing import NDArray

class CrossSequenceMemory:
    """Double-buffered LTM. Zero alloc. Field-based decay."""

    def __init__(self, n_heads: int, head_dim: int):
        size = n_heads * head_dim
        # Using float64 because FHRREngine operates on [-pi, pi] radians, requiring precision.
        self.ltm_a = np.zeros(size, dtype=np.float64)
        self.ltm_b = np.zeros(size, dtype=np.float64)
        self.active_is_a = True

        self.strength = np.full(n_heads, 0.5, dtype=np.float64)
        self.age = np.zeros(n_heads, dtype=np.uint32)

        self.lambda_ = 0.01
        self.consolidation_threshold = 0.3

    def read_active(self) -> NDArray[np.float64]:
        return self.ltm_a if self.active_is_a else self.ltm_b

    def write_ghost(self) -> NDArray[np.float64]:
        return self.ltm_b if self.active_is_a else self.ltm_a

    def swap(self):
        self.active_is_a = not self.active_is_a

    def reset(self):
        self.ltm_a.fill(0.0)
        self.ltm_b.fill(0.0)
        self.strength.fill(0.5)
        self.age.fill(0)
        self.active_is_a = True

    def inject_prior(self, external: NDArray[np.float64], gamma: float,
                     n_heads: int, head_dim: int):
        """Branchless blend: LTM <- gamma*external + (1-gamma)*LTM
        Note: Adapted for FHRR continuous phases."""
        ltm = self.write_ghost()
        one_minus_g = 1.0 - gamma

        # Fully vectorized, no loop over heads needed for the array operations
        ltm[:] = (gamma * external + one_minus_g * ltm) % (2 * np.pi)

        # Injection strengthens memory
        self.strength = np.clip(self.strength + 0.1, 0.0, 1.0)
        self.swap()

class AdaptiveFHRRAttention:
    """Adaptive FHRR Attention — WITH CROSS-SEQUENCE MEMORY"""

    def __init__(self, n_heads: int, total_dim: int, max_seq: int):
        assert total_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = total_dim // n_heads
        self.total_dim = total_dim
        self.max_seq = max_seq
        h, d, hd = self.n_heads, self.head_dim, self.total_dim

        # Short-term (volatile)
        self.stm_buf = np.zeros(h * d, dtype=np.float64)
        self.stm_momentum = np.zeros(h * d, dtype=np.float64)
        self.v_hat_buf = np.zeros(max_seq * h * d, dtype=np.float64)
        self.out_buf = np.zeros(max_seq * hd, dtype=np.float64)
        self.retrieval_error = np.zeros(h, dtype=np.float64)

        # Long-term (episodic, cross-sequence)
        self.memory = CrossSequenceMemory(n_heads, d)

        # Pre-allocated composite buffer (zero alloc di forward)
        self.composite = np.zeros(h * d, dtype=np.float64)

        # Generate frozen role vectors (seeded, deterministic)
        rng = 0x12345678
        self.role_vectors = np.zeros(h * d, dtype=np.float64)
        self.role_rotation = np.zeros(h, dtype=np.float64)

        # Random phase vectors for roles. In FHRR, binding is addition modulo 2pi.
        # No need to normalize length to 1.0 like the original Rust code.
        np.random.seed(42)
        for head in range(h):
            base = head * d
            self.role_vectors[base:base + d] = np.random.uniform(-np.pi, np.pi, d)

        # Adaptive Control
        self.plasticity = 0.1
        self.consolidation_rate = 0.05
        self.surprise_threshold = 0.6
        self.eps = 1e-15

    # (Forward method will be implemented in the next step to fully eliminate loops)

    def inject_collective_prior(self, collective: NDArray[np.float64], gamma: float):
        """Blackboard Injection: Collective Consciousness sebagai Prior"""
        self.memory.inject_prior(
            collective, gamma, self.n_heads, self.head_dim
        )

    def begin_new_session(self):
        """Session boundary: LTM di-preserve tapi strength di-decay"""
        self.memory.strength *= 0.5
        self.memory.age = np.maximum(0, self.memory.age.astype(np.int32) - 5).astype(np.uint32)

        self.stm_buf.fill(0.0)
        self.stm_momentum.fill(0.0)
        self.retrieval_error.fill(0.0)

    def hard_reset_memory(self):
        self.memory.reset()
        self.stm_buf.fill(0.0)
        self.stm_momentum.fill(0.0)
        self.retrieval_error.fill(0.0)
        self.role_rotation.fill(0.0)

    def set_plasticity(self, alpha: float):
        self.plasticity = max(0.001, min(0.99, alpha))

    def set_consolidation_rate(self, rate: float):
        self.consolidation_rate = max(0.001, min(0.5, rate))

    def forward(self, x_seq: NDArray[np.float64], seq_len: int) -> NDArray[np.float64]:
        """Vectorized Forward Pass without nested seq_len/head loops."""
        assert seq_len <= self.max_seq
        h, d, hd = self.n_heads, self.head_dim, self.total_dim

        # x_seq is (seq_len * hd). Reshape to (seq_len, h, d)
        x_reshaped = x_seq[:seq_len * hd].reshape(seq_len, h, d)
        role_reshaped = self.role_vectors.reshape(1, h, d) # (1, h, d)

        # =====================================================================
        # PHASE 0: DECAY LONG-TERM MEMORY
        # =====================================================================
        ltm_read = self.memory.read_active().copy()
        ghost = self.memory.write_ghost()

        # retrieval_error: (h,), strength: (h,)
        decay_factor = np.exp(
            -self.memory.lambda_ * self.retrieval_error * (1.5 - self.memory.strength)
        ) # (h,)
        # Broadcast decay across dimension d
        decay_factor_expanded = np.repeat(decay_factor, d) # (h*d,)

        # FHRR phases don't decay by scaling length, they decay by noise or moving towards zero.
        # But keeping with the algorithm's spirit: we scale the phase towards 0 (forgetting).
        ghost[:] = (ltm_read * decay_factor_expanded) % (2 * np.pi)
        self.memory.swap()

        ltm_active = self.memory.read_active()

        # =====================================================================
        # PHASE 1: BUILD SHORT-TERM BUNDLE (STM)
        # =====================================================================
        # FHRR Bind: x_t + role
        bind_res = (x_reshaped + role_reshaped) % (2 * np.pi) # (seq_len, h, d)

        # Bundle over sequence: bundle in FHRR is the angle of sum of complex exponentials
        complex_sum = np.sum(np.exp(1j * bind_res), axis=0) # (h, d)
        stm_new = np.angle(complex_sum) % (2 * np.pi) # (h, d)
        stm_new_flat = stm_new.flatten() # (h*d,)

        # Momentum smoothing: FHRR bundle of current STM and new STM
        # Convert to complex, scale by momentum weights, sum, and take angle
        mom_c = 0.8 * np.exp(1j * self.stm_momentum) + 0.2 * np.exp(1j * stm_new_flat)
        self.stm_buf[:] = np.angle(mom_c) % (2 * np.pi)
        self.stm_momentum[:] = self.stm_buf

        # =====================================================================
        # PHASE 2: COMPOSITE BUNDLE (LTM + STM)
        # =====================================================================
        ltm_gate = self.memory.strength * (1.0 - np.exp(-0.1 * self.memory.age)) # (h,)
        stm_gate = 1.0 - ltm_gate # (h,)

        ltm_gate_exp = np.repeat(ltm_gate, d) # (h*d,)
        stm_gate_exp = np.repeat(stm_gate, d) # (h*d,)

        # Complex FHRR bundle blending
        comp_c = ltm_gate_exp * np.exp(1j * ltm_active) + stm_gate_exp * np.exp(1j * self.stm_buf)
        self.composite[:] = np.angle(comp_c) % (2 * np.pi)

        # =====================================================================
        # PHASE 3: UNBIND & RETRIEVAL
        # =====================================================================
        comp_reshaped = self.composite.reshape(1, h, d)

        # Unbind FHRR: comp - role
        # v_hat = unbind(composite, role) = composite - role
        v_hat_reshaped = (comp_reshaped - role_reshaped) % (2 * np.pi) # (1, h, d)
        # Broadcast across seq_len
        v_hat_seq = np.broadcast_to(v_hat_reshaped, (seq_len, h, d)) # (seq_len, h, d)

        self.v_hat_buf[:seq_len * hd] = v_hat_seq.flatten()

        # FHRR Sim: cos(v_hat - x_t)
        # We want error = 1 - sim(v_hat, x)
        sim_per_t_h = np.mean(np.cos(v_hat_seq - x_reshaped), axis=2) # (seq_len, h)
        self.retrieval_error = np.mean(1.0 - np.abs(sim_per_t_h), axis=0) # (h,)

        # =====================================================================
        # PHASE 4: CONSOLIDATION (STM -> LTM)
        # =====================================================================
        ghost = self.memory.write_ghost()
        ltm_read = self.memory.read_active().copy()

        # Branchless gate: 1 if error <= threshold, else 0
        consolidate = np.where(self.retrieval_error <= self.memory.consolidation_threshold, 1.0, 0.0) # (h,)
        consolidate_exp = np.repeat(consolidate, d) # (h*d,)

        # Blend new_val
        new_val_c = (1.0 - self.consolidation_rate) * np.exp(1j * ltm_read) + self.consolidation_rate * np.exp(1j * self.stm_buf)
        new_val = np.angle(new_val_c) % (2 * np.pi)

        ghost[:] = np.where(consolidate_exp == 1.0, new_val, ltm_read)

        self.memory.strength = np.minimum(self.memory.strength + consolidate * 0.1, 1.0)
        self.memory.age += consolidate.astype(np.uint32)

        self.memory.swap()

        # =====================================================================
        # PHASE 5: MERGE HEADS + RESONANCE GATING
        # =====================================================================
        # Suppress heads with high error
        suppress = np.where(self.retrieval_error > self.surprise_threshold, 1.0, 0.0) # (h,)
        head_gates = 1.0 - suppress * 0.9 # (h,)
        head_gates_exp = head_gates.reshape(1, h, 1) # (1, h, 1)

        # Apply gates to v_hat_seq. In FHRR, gating is difficult.
        # A common proxy is to interpolate towards 0 (neutral phase) or attenuate amplitude in complex domain.
        # We will do a complex blend: gate * exp(1j * v_hat)
        out_c = head_gates_exp * np.exp(1j * v_hat_seq) # (seq_len, h, d)
        out_reshaped = np.angle(out_c) % (2 * np.pi)

        out_flat = out_reshaped.flatten()
        self.out_buf[:seq_len * hd] = out_flat

        # =====================================================================
        # PHASE 6: SURPRISE-DRIVEN EXPLORATION
        # =====================================================================
        avg_error = np.mean(self.retrieval_error)

        if avg_error > self.surprise_threshold * 0.8:
            self.role_rotation += 0.05

            # FHRR perturbation: Shift phase
            rot_exp = np.repeat(self.role_rotation, d)
            self.role_vectors = (self.role_vectors + rot_exp) % (2 * np.pi)

            self.stm_buf.fill(0.0)
            self.stm_momentum.fill(0.0)

        return self.out_buf[:seq_len * hd]

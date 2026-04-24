import numpy as np
from collections import defaultdict, deque
import time
from typing import Dict, List, Tuple, Optional, Any

class FHRREngine:

    def __init__(self, dim: int = 4096, n_hash_tables: int = 12, n_bins: int = 32, use_bpemb: bool = False):
        self.dim = dim
        self.use_bpemb = use_bpemb
        if use_bpemb:
            try:
                from bpemb import BPEmb
                self.bpemb = BPEmb(lang="id", vs=10000, dim=50)
                # Fixed random projection matrix for mapping 50d to `dim`
                np.random.seed(42)
                self.bpemb_proj = np.random.randn(50, dim) / np.sqrt(50)
            except ImportError:
                print("[Engine] Warning: bpemb is not installed. Disabling bpemb integration.")
                self.use_bpemb = False

        self.n_tables = n_hash_tables
        self.n_bins = n_bins
        self.token_names: List[str] = []
        self.token_phases: List[np.ndarray] = []
        self.token_categories: List[str] = []
        self.token_polarities: List[int] = []
        self._token_name_to_idx: Dict[str, int] = {}
        self.role_names: List[str] = []
        self.role_phases: List[np.ndarray] = []
        self._role_name_to_idx: Dict[str, int] = {}
        self.poles: Dict[str, Dict[str, np.ndarray]] = {}
        self.pole_categories: set = set()
        self.transforms: Dict[str, Dict[str, Any]] = {}
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.rule_counter: int = 0
        self.lsh_tables: List[Dict[int, List[int]]] = [{} for _ in range(n_hash_tables)]
        self.lsh_projections: List[np.ndarray] = []
        for _ in range(n_hash_tables):
            proj = np.random.normal(0, 1, dim)
            proj = proj / np.linalg.norm(proj)
            self.lsh_projections.append(proj)
        self.episodic_capacity: int = 5000
        self.episodic_buffer: List[Dict[str, Any]] = []
        self.episodic_head: int = 0

        # Vectorized buffers for episodic memory
        self._episodic_C_v: Optional[np.ndarray] = None
        self._episodic_S_v: Optional[np.ndarray] = None
        self._episodic_decay: Optional[np.ndarray] = None

        self._ws = np.zeros(dim)
        self._ws2 = np.zeros(dim)
        self._cbuf = np.zeros(dim, dtype=np.complex128)
        self.query_count = 0
        self.lsh_hits = 0
        self.linear_falls = 0
        # Training metrics
        self.learned_transforms_history: List[Dict] = []
        self.induced_rules_history: List[Dict] = []

    def alloc(self) -> np.ndarray:
        return np.random.uniform(-np.pi, np.pi, self.dim)

    def _get_bpemb_vector(self, name: str) -> Optional[np.ndarray]:
        if not self.use_bpemb:
            return None
        tokens = self.bpemb.encode(name)
        vecs = []
        for t in tokens:
            try:
                idx = self.bpemb.emb.key_to_index[t]
                vecs.append(self.bpemb.vectors[idx])
            except KeyError:
                continue
        if not vecs:
            return None
        vec_50 = np.sum(vecs, axis=0) / len(vecs)
        # Project and wrap
        vec_hrr = np.dot(vec_50, self.bpemb_proj)
        # scale so variance covers [-pi, pi] reasonably
        vec_hrr = (vec_hrr * np.pi) % (2 * np.pi)
        return vec_hrr


    def bind(self, a: np.ndarray, b: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        if out is None:
            out = self._ws
        np.add(a, b, out=out)
        np.mod(out, 2 * np.pi, out=out)
        return out

    def unbind(self, bound: np.ndarray, probe: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        if out is None:
            out = self._ws
        np.subtract(bound, probe, out=out)
        np.mod(out, 2 * np.pi, out=out)
        return out

    def bundle(self, vectors: List[np.ndarray], weights: Optional[np.ndarray] = None,
               out: Optional[np.ndarray] = None) -> np.ndarray:
        if out is None:
            out = self._cbuf
        out.fill(0)
        if weights is None:
            weights = np.ones(len(vectors)) / len(vectors)
        for v, w in zip(vectors, weights):
            out += w * np.exp(1j * v)
        return np.angle(out)

    def sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.cos(a - b)))

    def _lsh_hash(self, vec: np.ndarray, table_idx: int) -> int:
        proj = self.lsh_projections[table_idx]
        scalar = np.dot(np.cos(vec), proj)
        bin_idx = int((scalar + 1.0) / 2.0 * self.n_bins)
        return max(0, min(bin_idx, self.n_bins - 1))

    def define_poles(self, category: str) -> Dict[str, np.ndarray]:
        if category in self.poles:
            return self.poles[category]
        pole_pos = self.alloc()
        pole_neg = (pole_pos + np.pi) % (2 * np.pi)
        self.poles[category] = {'positive': pole_pos, 'negative': pole_neg}
        self.pole_categories.add(category)
        return self.poles[category]

    def add_token(self, name: str, category: str, polarity: int = 0,
                  prototype: Optional[np.ndarray] = None, noise: float = 0.20) -> np.ndarray:
        if name in self._token_name_to_idx:
            return self.token_phases[self._token_name_to_idx[name]]

        bpemb_vec = self._get_bpemb_vector(name)

        if bpemb_vec is not None:
            vec = bpemb_vec
        else:
            if prototype is None:
                prototype = self.alloc()
            vec = (prototype + np.random.normal(0, noise, self.dim)) % (2 * np.pi)
            if polarity != 0 and category in self.poles:
                pole_vec = self.poles[category]['positive'] if polarity > 0 else self.poles[category]['negative']
                vec = (vec + pole_vec * 0.4) % (2 * np.pi)

        idx = len(self.token_names)
        self.token_names.append(name)
        self.token_phases.append(vec)
        self.token_categories.append(category)
        self._token_name_to_idx[name] = idx
        self.token_polarities.append(polarity)
        for t in range(self.n_tables):
            bin_idx = self._lsh_hash(vec, t)
            if bin_idx not in self.lsh_tables[t]:
                self.lsh_tables[t][bin_idx] = []
            self.lsh_tables[t][bin_idx].append(idx)
        return vec

    def get_token_idx(self, name: str) -> Optional[int]:
        return self._token_name_to_idx.get(name)

    def get_token(self, name: str) -> Optional[np.ndarray]:
        idx = self._token_name_to_idx.get(name)
        if idx is not None:
            return self.token_phases[idx]
        return None

    def add_role(self, name: str) -> np.ndarray:
        if name in self._role_name_to_idx:
            return self.role_phases[self._role_name_to_idx[name]]
        vec = self.alloc()
        idx = len(self.role_names)
        self.role_names.append(name)
        self.role_phases.append(vec)
        self._role_name_to_idx[name] = idx
        return vec

    def get_role(self, name: str) -> Optional[np.ndarray]:
        idx = self._role_name_to_idx.get(name)
        if idx is not None:
            return self.role_phases[idx]
        return None

    def encode(self, bindings: Dict[str, str]) -> Optional[np.ndarray]:
        self._cbuf.fill(0)
        count = 0
        for role_name, token_name in bindings.items():
            role_vec = self.get_role(role_name)
            token_vec = self.get_token(token_name)
            if role_vec is None or token_vec is None:
                continue
            # Zero-allocation Phase FHRR binding: accumulation inline.
            # np.exp handles the modulo boundary natively on the complex plane.
            self._cbuf += np.exp(1j * (role_vec + token_vec))
            count += 1

        if count == 0:
            return None

        # Extract phase as the final FHRR bundled structure
        return np.angle(self._cbuf) % (2 * np.pi)

    def decode(self, struct_vec: np.ndarray, threshold: float = 0.40) -> Dict[str, Tuple[str, float]]:
        if not self.role_names or not self.token_names:
            return {}

        # Vectorized decoding for massive speedup over O(M * N) python loops
        # 1. Unbind struct with all roles via broadcasting
        # struct_vec: (dim,), role_phases: (M, dim) -> unbound_mat: (M, dim)
        roles_mat = np.array(self.role_phases)
        unbound_mat = (struct_vec - roles_mat) % (2 * np.pi)

        # 2. Extract all token phases
        tokens_mat = np.array(self.token_phases) # (N, dim)

        # 3. Vectorized cosine similarity using cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
        C_u = np.cos(unbound_mat) # (M, dim)
        S_u = np.sin(unbound_mat) # (M, dim)
        C_t = np.cos(tokens_mat)  # (N, dim)
        S_t = np.sin(tokens_mat)  # (N, dim)

        # sim_mat shape: (M, N)
        sim_mat = (C_u @ C_t.T + S_u @ S_t.T) / self.dim

        decomposition = {}
        # Max along the tokens axis for each role
        best_indices = np.argmax(sim_mat, axis=1)
        best_sims = np.max(sim_mat, axis=1)

        for m, role_name in enumerate(self.role_names):
            sim = float(best_sims[m])
            if sim > threshold:
                decomposition[role_name] = (self.token_names[best_indices[m]], sim)

        return decomposition

    def cleanup(self, query_vec: np.ndarray, threshold: float = 0.45,
                probe_factor: int = 2) -> Tuple[Optional[str], float]:
        self.query_count += 1
        candidates = set()
        for t in range(self.n_tables):
            bin_idx = self._lsh_hash(query_vec, t)
            if bin_idx in self.lsh_tables[t]:
                candidates.update(self.lsh_tables[t][bin_idx])
        if len(candidates) < probe_factor * 3:
            for t in range(self.n_tables):
                bin_idx = self._lsh_hash(query_vec, t)
                for delta in [-1, 1]:
                    neighbor = bin_idx + delta
                    if 0 <= neighbor < self.n_bins and neighbor in self.lsh_tables[t]:
                        candidates.update(self.lsh_tables[t][neighbor])
        if len(candidates) < 5:
            self.linear_falls += 1
            candidates = range(len(self.token_names))
        else:
            self.lsh_hits += 1
        best_name = None
        best_sim = -1.0
        for idx in candidates:
            sim = self.sim(query_vec, self.token_phases[idx])
            if sim > best_sim:
                best_sim = sim
                best_name = self.token_names[idx]
        if best_sim > threshold:
            return best_name, best_sim
        return None, best_sim

    def kernel_query(self, query_vec: np.ndarray, radius: float = 0.3,
                     top_k: int = 5) -> List[Tuple[str, float, float]]:
        results = []
        for idx, vec in enumerate(self.token_phases):
            sim = self.sim(query_vec, vec)
            weight = np.exp(-((1.0 - sim) ** 2) / (2 * radius ** 2))
            if weight > 0.1:
                results.append((self.token_names[idx], sim, weight))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def store_episodic(self, vec: np.ndarray, metadata: Optional[Dict] = None):
        entry = {
            'vector': vec.copy(),
            'timestamp': time.time(),
            'access_count': 0,
            'decay_factor': 1.0,
            'metadata': metadata or {}
        }
        if len(self.episodic_buffer) >= self.episodic_capacity:
            self.episodic_buffer[self.episodic_head] = entry
            self.episodic_head = (self.episodic_head + 1) % self.episodic_capacity
        else:
            self.episodic_buffer.append(entry)

        # Invalidate the vectorized buffer cache
        self._episodic_C_v = None
        self._episodic_S_v = None
        self._episodic_decay = None

    def apply_decay(self, current_time: Optional[float] = None, decay_rate: float = 0.001):
        if current_time is None:
            current_time = time.time()
        for entry in self.episodic_buffer:
            age = current_time - entry['timestamp']
            retention = np.exp(-decay_rate * age) * (1 + 0.1 * entry['access_count'])
            entry['decay_factor'] = retention
            if retention < 0.9:
                noise_level = (1.0 - retention) * 0.1
                entry['vector'] = (entry['vector'] +
                                  np.random.normal(0, noise_level, self.dim)) % (2 * np.pi)

    def query_episodic(self, query_vec: np.ndarray, threshold: float = 0.4):
        if not self.episodic_buffer:
            return None, -1.0

        # Vectorized episodic memory search with maintained buffer
        if self._episodic_C_v is None or len(self._episodic_C_v) != len(self.episodic_buffer):
            vectors = np.array([e['vector'] for e in self.episodic_buffer])
            self._episodic_C_v = np.cos(vectors)
            self._episodic_S_v = np.sin(vectors)
            self._episodic_decay = np.array([e['decay_factor'] for e in self.episodic_buffer])
        else:
            # Refresh decay factors in case apply_decay was called
            # To be 100% safe, we can just rebuild it or selectively update
            self._episodic_decay = np.array([e['decay_factor'] for e in self.episodic_buffer])

        C_q = np.cos(query_vec)
        S_q = np.sin(query_vec)

        # sim_arr shape: (N,)
        sim_arr = (self._episodic_C_v @ C_q + self._episodic_S_v @ S_q) / self.dim
        compensated = sim_arr * self._episodic_decay

        best_idx = np.argmax(compensated)
        best_score = float(compensated[best_idx])

        if best_score > threshold:
            best_match = self.episodic_buffer[best_idx]
            best_match['access_count'] += 1
            return best_match, best_score
        return None, best_score

    def learn_transform(self, name: str, source_vec: np.ndarray,
                        target_vec: np.ndarray, min_confidence: float = 0.5) -> Tuple[Optional[str], float]:
        diff = (target_vec - source_vec + np.pi) % (2 * np.pi) - np.pi
        test = (source_vec + diff) % (2 * np.pi)
        conf = self.sim(test, target_vec)
        if conf < min_confidence:
            return None, conf
        self.transforms[name] = {'vector': diff, 'confidence': conf, 'usage_count': 0,
                                 'source': None, 'target': None}  # akan diisi oleh trainer
        return name, conf

    def apply_transform(self, source_vec: np.ndarray, transform_name: str,
                        scale: float = 1.0) -> Optional[np.ndarray]:
        if transform_name not in self.transforms:
            return None
        t = self.transforms[transform_name]
        result = (source_vec + t['vector'] * scale) % (2 * np.pi)
        t['usage_count'] += 1
        return result

    def chain_transforms(self, source_vec: np.ndarray,
                         transform_names: List[str]) -> Tuple[Optional[np.ndarray], float]:
        current = source_vec.copy()
        total_conf = 1.0
        for tname in transform_names:
            if tname not in self.transforms:
                return None, 0.0
            t = self.transforms[tname]
            current = (current + t['vector']) % (2 * np.pi)
            total_conf *= t['confidence']
        return current, total_conf

    def add_rule(self, pattern_bindings: Dict[str, str], action: str,
                 transform_name: Optional[str] = None, confidence: float = 0.7,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        pattern_vec = self.encode(pattern_bindings)
        if pattern_vec is None:
            raise ValueError("Invalid pattern bindings")
        rid = f"rule_{self.rule_counter}"
        self.rule_counter += 1
        self.rules[rid] = {
            'pattern': pattern_vec, 'action': action, 'confidence': confidence,
            'support': 1, 'bindings': pattern_bindings, 'transform': transform_name,
            'metadata': metadata or {}
        }
        self.induced_rules_history.append({
            'id': rid, 'bindings': pattern_bindings, 'action': action,
            'confidence': confidence, 'timestamp': time.time(), 'metadata': metadata or {}
        })
        return rid

    def match_rule(self, query_vec: np.ndarray, threshold: float = 0.65,
                   metadata_filter: Optional[Dict[str, Any]] = None):
        best_rid = None
        best_sim = -1.0
        for rid, rule in self.rules.items():
            # Apply metadata filter if provided
            if metadata_filter:
                match = True
                for k, v in metadata_filter.items():
                    if rule['metadata'].get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            sim = self.sim(query_vec, rule['pattern'])
            if sim > best_sim:
                best_sim = sim
                best_rid = rid
        if best_rid and best_sim > threshold:
            return self.rules[best_rid], best_sim
        return None, best_sim

    def detect_contradiction(self, vec1: np.ndarray, vec2: np.ndarray,
                             threshold: float = 0.0) -> Tuple[bool, List[Dict]]:
        decomp1 = self.decode(vec1, threshold=0.35)
        decomp2 = self.decode(vec2, threshold=0.35)
        conflicts = []
        for role in set(decomp1.keys()) & set(decomp2.keys()):
            f1, c1 = decomp1[role]
            f2, c2 = decomp2[role]
            v1 = self.get_token(f1)
            v2 = self.get_token(f2)
            if v1 is None or v2 is None:
                continue
            sim = self.sim(v1, v2)
            if sim < threshold:
                conflicts.append({
                    'role': role, 'filler1': f1, 'filler2': f2,
                    'filler_sim': sim, 'severity': abs(sim)
                })
        return len(conflicts) > 0, conflicts

    # -------------------------------------------------------------------------
    # DATASET INTEGRATION (v2)
    # -------------------------------------------------------------------------

    def build_from_dataset(self, dataset: Dict[str, Any]):
        vocab = dataset.get('vocab', {})
        cat_map = vocab.get('categories', {})
        pole_map = vocab.get('poles', {})

        for pole_cat in pole_map.keys():
            self.define_poles(pole_cat)

        prototypes = {cat: self.alloc() for cat in cat_map.keys()}

        for cat_name, tokens in cat_map.items():
            proto = prototypes[cat_name]
            pole_cat = cat_name if cat_name in pole_map else None
            for tok in tokens:
                polarity = 0
                if pole_cat:
                    if tok in pole_map[pole_cat].get('positive', []):
                        polarity = +1
                    elif tok in pole_map[pole_cat].get('negative', []):
                        polarity = -1
                self.add_token(tok, cat_name, polarity, proto,
                               noise=0.18 if polarity != 0 else 0.22)

        # Roles dari dataset (auto-detect dari observations + qa_pairs)
        all_roles = set()
        for obs in dataset.get('observations', []):
            all_roles.update(obs.get('bindings', {}).keys())
        for qa in dataset.get('qa_pairs', []):
            all_roles.add(qa.get('answer_role'))
        for r in sorted(all_roles):
            if r:
                self.add_role(r)

        print(f"[Engine] Vocab: {len(self.token_names)} tokens, {len(self.role_names)} roles")

    def learn_transform_from_data(self, name: str, from_token: str, to_token: str,
                                  min_confidence: float = 0.5) -> Tuple[Optional[str], float]:
        src = self.get_token(from_token)
        tgt = self.get_token(to_token)
        if src is None or tgt is None:
            return None, 0.0
        result, conf = self.learn_transform(name, src, tgt, min_confidence)
        if result:
            self.transforms[name]['source'] = from_token
            self.transforms[name]['target'] = to_token
            self.learned_transforms_history.append({
                'name': name, 'from': from_token, 'to': to_token,
                'confidence': conf, 'timestamp': time.time()
            })
        return result, conf
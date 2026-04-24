# =============================================================================
# AGENT LAYER: Self-Supervised Discovery
# =============================================================================
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import re

class SelfSupervisedDiscovery:
    def __init__(self, engine, window_size: int = 3,
                 min_cooccurrence_freq: int = 2,
                 relation_sim_threshold: float = 0.40,
                 role_entropy_threshold: float = 0.8):
        self.engine = engine
        self.window = window_size
        self.min_freq = min_cooccurrence_freq
        self.rel_sim_threshold = relation_sim_threshold
        self.role_entropy_threshold = role_entropy_threshold

        self.sentences: List[List[str]] = []
        self.token_positions: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.cooccurrence: Dict[Tuple[str, str], int] = Counter()
        self.positional_profiles: Dict[str, np.ndarray] = {}

        self.discovered_roles: Dict[str, Dict] = {}
        self.discovered_relations: Dict[str, Dict] = {}
        self.discovered_patterns: List[Dict] = []
        self._ws = np.zeros(engine.dim)

    def ingest_corpus(self, raw_sentences: List[str]):
        self.sentences = []
        for i, sent in enumerate(raw_sentences):
            tokens = [t for t in re.findall(r'\b\w+\b', sent.lower())]
            if len(tokens) < 2:
                continue
            self.sentences.append(tokens)

            for j, tok in enumerate(tokens):
                if self.engine.get_token(tok) is None:
                    continue
                self.token_positions[tok].append((i, j))
        print(f"[Discovery] Corpus: {len(self.sentences)} sentences, {sum(len(s) for s in self.sentences)} tokens")

    def induce_roles(self) -> Dict[str, str]:
        if not self.sentences:
            return {}

        max_len = max(len(s) for s in self.sentences)
        profiles = {}

        for tok, positions in self.token_positions.items():
            hist = np.zeros(max_len)
            for sent_idx, pos in positions:
                hist[pos] += 1
            hist = hist / (hist.sum() + 1e-12)
            profiles[tok] = hist

        tokens = list(profiles.keys())
        n = len(tokens)
        if n < 3:
            return {}

        aff = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = profiles[tokens[i]], profiles[tokens[j]]
                L = max(len(p1), len(p2))
                v1 = np.pad(p1, (0, L - len(p1)))
                v2 = np.pad(p2, (0, L - len(p2)))
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
                aff[i, j] = sim
                aff[j, i] = sim

        centroids = np.array([
            self._ideal_profile(max_len, 'start'),
            self._ideal_profile(max_len, 'middle'),
            self._ideal_profile(max_len, 'end')
        ])

        # Pre-build normalized feature matrix X
        X = np.zeros((n, max_len))
        for i, tok in enumerate(tokens):
            p = profiles[tok]
            X[i, :len(p)] = p

        # Normalize rows of X
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / (norms + 1e-12)

        labels = np.zeros(n, dtype=int)
        for _ in range(15):
            # Normalize centroids
            c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            centroids_norm = centroids / (c_norms + 1e-12)

            # Vectorized dot product for cosine similarity
            sims = X @ centroids_norm.T
            labels = np.argmax(sims, axis=1)

            # Vectorized centroid update
            for k in range(3):
                mask = labels == k
                if mask.any():
                    avg = X[mask].mean(axis=0)
                    centroids[k] = avg / (avg.sum() + 1e-12)

        role_names = ['discovered_subject', 'discovered_predicate', 'discovered_object']
        role_map = {}

        # Calculate final confidence dynamically
        c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids_norm = centroids / (c_norms + 1e-12)
        final_sims = X @ centroids_norm.T
        confidences = np.max(final_sims, axis=1)

        for i, tok in enumerate(tokens):
            assigned_role = role_names[labels[i]]
            role_map[tok] = assigned_role
            if assigned_role not in self.engine.role_names:
                self.engine.add_role(assigned_role)
            self.discovered_roles[tok] = {
                'role': assigned_role,
                'position_profile': profiles[tok],
                'confidence': float(confidences[i])
            }

        print(f"[Discovery] Roles induced: {len(set(role_map.values()))} roles for {len(role_map)} tokens")
        return role_map

    def _ideal_profile(self, length: int, mode: str) -> np.ndarray:
        p = np.zeros(length)
        if mode == 'start':
            p[:max(1, length // 3)] = 1.0
        elif mode == 'middle':
            mid = length // 2
            p[max(0, mid - 1):min(length, mid + 2)] = 1.0
        elif mode == 'end':
            p[-max(1, length // 3):] = 1.0
        return p / (p.sum() + 1e-12)

    def mine_cooccurrence(self):
        # Pre-cache whether a token is valid to avoid calling get_token inside tight loops
        valid_tokens = set(self.engine.token_names)
        freq_cache = {t: self._token_freq(t) for t in valid_tokens}

        for sent in self.sentences:
            n = len(sent)
            for i in range(n):
                t1 = sent[i]
                if t1 not in valid_tokens:
                    continue
                f1 = freq_cache.get(t1, 0)

                for j in range(i + 1, min(n, i + self.window + 1)):
                    t2 = sent[j]
                    if t2 not in valid_tokens:
                        continue
                    f2 = freq_cache.get(t2, 0)

                    if f1 > f2:
                        self.cooccurrence[(t2, t1)] += 1
                    else:
                        self.cooccurrence[(t1, t2)] += 1
        print(f"[Discovery] Co-occurrence pairs: {len(self.cooccurrence)}")

    def _token_freq(self, token: str) -> int:
        return len(self.token_positions.get(token, []))

    def induce_relations(self) -> Dict[str, Dict]:
        relations = {}
        for (t1, t2), freq in self.cooccurrence.items():
            if freq < self.min_freq:
                continue

            v1 = self.engine.get_token(t1)
            v2 = self.engine.get_token(t2)
            if v1 is None or v2 is None:
                continue

            sim = self.engine.sim(v1, v2)
            cat1 = self.engine.token_categories[self.engine._token_name_to_idx[t1]]
            cat2 = self.engine.token_categories[self.engine._token_name_to_idx[t2]]

            if 0.15 < sim < 0.65 and cat1 != cat2:
                rel_name = f"rel_{t1}_to_{t2}"
                diff = (v2 - v1 + np.pi) % (2 * np.pi) - np.pi
                conf = (0.5 + 0.5 * (1 - abs(sim))) * min(1.0, freq / 5.0)

                self.engine.transforms[rel_name] = {
                    'vector': diff, 'confidence': float(conf), 'usage_count': 0,
                    'source': t1, 'target': t2, 'auto_discovered': True, 'frequency': freq
                }
                relations[rel_name] = {
                    'from': t1, 'to': t2, 'similarity': float(sim),
                    'confidence': float(conf), 'frequency': freq
                }

        self.discovered_relations = relations
        print(f"[Discovery] Relations induced: {len(relations)}")
        return relations

    def discover_patterns(self) -> List[Dict]:
        if not self.discovered_roles:
            self.induce_roles()

        patterns = Counter()
        for sent in self.sentences:
            roles = []
            for tok in sent:
                if tok in self.discovered_roles:
                    roles.append(self.discovered_roles[tok]['role'])
                else:
                    roles.append('?')
            for n in [3, 4]:
                for i in range(len(roles) - n + 1):
                    gram = tuple(roles[i:i + n])
                    patterns[gram] += 1

        frequent = [(p, c) for p, c in patterns.items() if c >= self.min_freq]
        frequent.sort(key=lambda x: -x[1])

        self.discovered_patterns = [
            {'pattern': p, 'frequency': c, 'length': len(p)}
            for p, c in frequent[:20]
        ]
        print(f"[Discovery] Patterns discovered: {len(self.discovered_patterns)}")
        return self.discovered_patterns

    def discover_all(self, raw_corpus: List[str]) -> Dict[str, Any]:
        self.ingest_corpus(raw_corpus)
        self.mine_cooccurrence()
        roles = self.induce_roles()
        relations = self.induce_relations()
        patterns = self.discover_patterns()

        return {
            'roles_discovered': len(roles),
            'relations_discovered': len(relations),
            'patterns_discovered': len(patterns),
            'role_map': roles,
            'relations': relations,
            'patterns': patterns
        }

    def query_relation_path(self, start_token: str, end_token: str, max_hops: int = 3) -> List[Tuple[str, float]]:
        if self.engine.get_token(start_token) is None:
            return []

        adj = defaultdict(list)
        for rel_name, info in self.discovered_relations.items():
            adj[info['from']].append((info['to'], rel_name, info['confidence']))

        visited = set()
        queue = [(start_token, 1.0, [start_token])]
        results = []

        while queue and len(results) < 10:
            current, conf, path = queue.pop(0)
            if current == end_token and len(path) > 1:
                results.append((path, conf))
                continue
            if len(path) > max_hops:
                continue
            for nxt, rel, rconf in adj.get(current, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, conf * rconf, path + [nxt]))

        return results

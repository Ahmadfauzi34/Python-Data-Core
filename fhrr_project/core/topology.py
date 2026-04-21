# =============================================================================
# TOPOLOGY LAYER: TDA, Sheaf, Fiber Bundle, MERA, Spectral
# =============================================================================
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from scipy.linalg import eigh, svd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# -----------------------------------------------------------------------------
# 1. Vietoris-Rips Persistent Homology (Z2)
# -----------------------------------------------------------------------------
@dataclass
class PersistencePair:
    dim: int
    birth: float
    death: float
    creator: Tuple[int, ...] = ()
    destroyer: Tuple[int, ...] = ()

class VietorisRipsPH:
    def __init__(self, max_dim: int = 1, k_neighbors: int = 12, max_dist: float = 1.5):
        self.max_dim = max_dim
        self.k = k_neighbors
        self.max_dist = max_dist
        self.simplices: List[Tuple[float, int, Tuple[int, ...]]] = []
        self.pairs: List[PersistencePair] = []
        self.positive: List[PersistencePair] = []
        self._index_map: Dict[Tuple[int, ...], int] = {}
        self._reduced: List[Set[int]] = []

    def fit(self, dist_matrix: np.ndarray):
        n = dist_matrix.shape[0]
        neighbors = self._knn_graph(dist_matrix, n)
        self._enumerate_simplices(dist_matrix, neighbors, n)
        self._reduce_boundary_matrix()
        return self

    def _knn_graph(self, D: np.ndarray, n: int) -> List[np.ndarray]:
        neigh = []
        for i in range(n):
            idx = np.argsort(D[i])[:self.k + 1]
            idx = idx[idx != i][:self.k]
            neigh.append(idx)
        return neigh

    def _enumerate_simplices(self, D: np.ndarray, neighbors: List[np.ndarray], n: int):
        simplices = []
        for i in range(n):
            simplices.append((0.0, 0, (i,)))
        edges = set()
        for i in range(n):
            for j in neighbors[i]:
                if i < j:
                    d = D[i, j]
                    if d <= self.max_dist:
                        edges.add((i, j))
                        simplices.append((float(d), 1, (i, j)))
        if self.max_dim >= 2:
            for i in range(n):
                nei = list(neighbors[i])
                for a in range(len(nei)):
                    for b in range(a + 1, len(nei)):
                        j, k = nei[a], nei[b]
                        if i < j and i < k and (j, k) in edges:
                            d = max(D[i, j], D[i, k], D[j, k])
                            if d <= self.max_dist:
                                simplices.append((float(d), 2, (i, j, k)))
        simplices.sort(key=lambda x: (x[0], x[1]))
        self.simplices = simplices
        self._index_map = {simp: idx for idx, (_, _, simp) in enumerate(simplices)}

    def _boundary(self, simplex: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        return [simplex[:i] + simplex[i+1:] for i in range(len(simplex))]

    def _reduce_boundary_matrix(self):
        cols = []
        for _, dim, simp in self.simplices:
            if dim == 0:
                cols.append(set())
            else:
                bdry = set()
                for face in self._boundary(simp):
                    if face in self._index_map:
                        bdry.add(self._index_map[face])
                cols.append(bdry)
        reduced = [set(c) for c in cols]
        low = {}
        self.pairs = []
        self.positive = []
        for j in range(len(reduced)):
            col = reduced[j]
            while col:
                l = max(col)
                if l in low:
                    i = low[l]
                    col = col.symmetric_difference(reduced[i])
                else:
                    low[l] = j
                    birth_val, dim_l, creator = self.simplices[l]
                    death_val, _, destroyer = self.simplices[j]
                    self.pairs.append(PersistencePair(
                        dim=dim_l, birth=birth_val, death=death_val,
                        creator=creator, destroyer=destroyer
                    ))
                    break
            if not col:
                birth_val, dim_j, creator = self.simplices[j]
                self.positive.append(PersistencePair(
                    dim=dim_j, birth=birth_val, death=float('inf'),
                    creator=creator
                ))
        self._reduced = reduced

    def betti(self, threshold: float) -> Dict[int, int]:
        alive = defaultdict(int)
        for p in self.pairs:
            if p.birth <= threshold < p.death:
                alive[p.dim] += 1
        for p in self.positive:
            if p.birth <= threshold:
                alive[p.dim] += 1
        return dict(alive)

    def persistence_diagram(self):
        births, deaths, dims = [], [], []
        for p in self.pairs + self.positive:
            births.append(p.birth)
            deaths.append(p.death if p.death != float('inf') else max(births) * 1.1 if births else 1.0)
            dims.append(p.dim)
        return np.array(births), np.array(deaths), np.array(dims)

    def significant_features(self, min_persistence: float = 0.1) -> List[PersistencePair]:
        out = []
        for p in self.pairs:
            if p.death - p.birth > min_persistence:
                out.append(p)
        for p in self.positive:
            if p.death == float('inf'):
                out.append(p)
        return out


# -----------------------------------------------------------------------------
# 2. Sheaf over Semantic Category Poset
# -----------------------------------------------------------------------------
@dataclass
class Stalk:
    category: str
    indices: List[int]
    center: np.ndarray
    principal: np.ndarray
    eigenvalues: np.ndarray

class SheafVSA:
    def __init__(self, engine):
        self.engine = engine
        self.stalks: Dict[str, Stalk] = {}
        self.restriction: Dict[Tuple[str, str], np.ndarray] = {}
        self.base_adj: Dict[str, Set[str]] = defaultdict(set)

    def build_stalks(self):
        cats = sorted(set(self.engine.token_categories))
        for cat in cats:
            idxs = [i for i, c in enumerate(self.engine.token_categories) if c == cat]
            vecs = np.stack([self.engine.token_phases[i] for i in idxs])
            center = self.engine.bundle(list(vecs))
            centered = np.exp(1j * (vecs - center))
            cov = centered @ centered.conj().T
            e, v = np.linalg.eigh(cov)
            k = min(5, len(idxs))
            self.stalks[cat] = Stalk(category=cat, indices=idxs, center=center,
                                     principal=v[:, -k:], eigenvalues=e[-k:])

    def build_base_space(self, cooccurrence_threshold: float = 0.15):
        cats = list(self.stalks.keys())
        n = len(cats)
        for i in range(n):
            for j in range(i + 1, n):
                c1, c2 = cats[i], cats[j]
                sim = self.engine.sim(self.stalks[c1].center, self.stalks[c2].center)
                if sim > cooccurrence_threshold:
                    self.base_adj[c1].add(c2)
                    self.base_adj[c2].add(c1)

    def compute_restriction(self, cat1: str, cat2: str) -> np.ndarray:
        c1 = self.stalks[cat1].center
        c2 = self.stalks[cat2].center
        transport = (c2 - c1 + np.pi) % (2 * np.pi) - np.pi
        self.restriction[(cat1, cat2)] = transport
        return transport

    def compute_all_restrictions(self):
        for c1 in self.base_adj:
            for c2 in self.base_adj[c1]:
                if (c1, c2) not in self.restriction:
                    self.compute_restriction(c1, c2)

    def restrict(self, vec: np.ndarray, cat1: str, cat2: str) -> np.ndarray:
        if (cat1, cat2) not in self.restriction:
            self.compute_restriction(cat1, cat2)
        return (vec + self.restriction[(cat1, cat2)]) % (2 * np.pi)

    def global_section_consistency(self, assignment: Dict[str, np.ndarray], tol: float = 0.35):
        violations = []
        for c1 in self.base_adj:
            for c2 in self.base_adj[c1]:
                if c1 not in assignment or c2 not in assignment:
                    continue
                v1r = self.restrict(assignment[c1], c1, c2)
                sim = self.engine.sim(v1r, assignment[c2])
                if sim < tol:
                    violations.append({'edge': (c1, c2), 'similarity': float(sim), 'severity': 1.0 - sim})
        return len(violations) == 0, violations

    def sheaf_cohomology_h0_estimate(self) -> int:
        n_cats = len(self.stalks)
        cats = list(self.stalks.keys())
        idx_map = {c: i for i, c in enumerate(cats)}
        adj = np.zeros((n_cats, n_cats))
        for c1 in self.base_adj:
            for c2 in self.base_adj[c1]:
                if (c1, c2) in self.restriction:
                    trans = self.restriction[(c1, c2)]
                    coherence = float(np.mean(np.cos(trans)))
                    if coherence > 0.5:
                        i, j = idx_map[c1], idx_map[c2]
                        adj[i, j] = 1
        n_comp, _ = connected_components(csgraph=csr_matrix(adj), directed=False)
        return int(n_comp)


# -----------------------------------------------------------------------------
# 3. Fiber Bundle (Category Base × Phase Fiber)
# -----------------------------------------------------------------------------
class FiberBundleVSA:
    def __init__(self, engine):
        self.engine = engine
        self.fibers: Dict[str, np.ndarray] = {}
        self.connections: Dict[Tuple[str, str], np.ndarray] = {}
        self.base_paths: Dict[Tuple[str, str], List[str]] = {}

    def build_fibers(self):
        cats = sorted(set(self.engine.token_categories))
        for cat in cats:
            idxs = [i for i, c in enumerate(self.engine.token_categories) if c == cat]
            self.fibers[cat] = np.stack([self.engine.token_phases[i] for i in idxs])

    def build_connections(self, sheaf):
        for (c1, c2), transport in sheaf.restriction.items():
            self.connections[(c1, c2)] = transport.copy()

    def parallel_transport(self, vec: np.ndarray, path: List[str]) -> np.ndarray:
        current = vec.copy()
        for i in range(len(path) - 1):
            c1, c2 = path[i], path[i + 1]
            if (c1, c2) in self.connections:
                current = (current + self.connections[(c1, c2)]) % (2 * np.pi)
        return current

    def compute_geodesic(self, cat1: str, cat2: str, sheaf) -> List[str]:
        if not self.base_paths:
            self._precompute_paths(sheaf)
        return self.base_paths.get((cat1, cat2), [cat1, cat2])

    def _precompute_paths(self, sheaf):
        cats = list(sheaf.stalks.keys())
        n = len(cats)
        idx = {c: i for i, c in enumerate(cats)}
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0)
        for c1 in sheaf.base_adj:
            for c2 in sheaf.base_adj[c1]:
                i, j = idx[c1], idx[c2]
                dist[i, j] = 1.0
        pred = np.full((n, n), -1, dtype=int)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
                        pred[i, j] = k
        def path(i, j):
            if pred[i, j] == -1:
                return [cats[i], cats[j]]
            k = pred[i, j]
            return path(i, k)[:-1] + path(k, j)
        for i in range(n):
            for j in range(n):
                if i != j and dist[i, j] < np.inf:
                    self.base_paths[(cats[i], cats[j])] = path(i, j)

    def curvature(self, loop: List[str], test_vec: np.ndarray) -> float:
        transported = self.parallel_transport(test_vec, loop + [loop[0]])
        diff = (transported - test_vec + np.pi) % (2 * np.pi) - np.pi
        return float(np.mean(np.abs(diff)))

    def section(self, category: str, token_name: Optional[str] = None) -> np.ndarray:
        if token_name is None:
            return self.engine.bundle(list(self.fibers[category]))
        idx = self.engine.token_names.index(token_name)
        return self.engine.token_phases[idx]


# -----------------------------------------------------------------------------
# 4. MERA Tensor Network Coarse-Graining
# -----------------------------------------------------------------------------
@dataclass
class MERALevel:
    disentanglers: List[np.ndarray] = field(default_factory=list)
    isometries: List[np.ndarray] = field(default_factory=list)

class MERAHierarchy:
    def __init__(self, engine, block_size: int = 64):
        self.engine = engine
        self.block_size = block_size
        assert engine.dim % block_size == 0, "dim must be divisible by block_size"
        self.n_blocks = engine.dim // block_size
        self.levels: List[MERALevel] = []

    def _reshape_blocks(self, vec: np.ndarray) -> np.ndarray:
        return np.exp(1j * vec).reshape(self.n_blocks, self.block_size)

    def _flatten_blocks(self, mat: np.ndarray) -> np.ndarray:
        return np.angle(mat).flatten()

    def disentangle_pair(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Za = self._reshape_blocks(a)
        Zb = self._reshape_blocks(b)
        corr = Za @ Zb.conj().T
        U, s, Vh = svd(corr, full_matrices=False)
        disentangler = U @ Vh
        Za_prime = disentangler @ Za
        Zb_prime = disentangler.conj().T @ Zb
        return self._flatten_blocks(Za_prime), self._flatten_blocks(Zb_prime), s

    def isometric_compress(self, local: np.ndarray, entangled: np.ndarray,
                           spectrum: np.ndarray, retain_ratio: float = 0.5) -> np.ndarray:
        p = spectrum / (spectrum.sum() + 1e-12)
        entropy = -np.sum(p * np.log(p + 1e-12))
        weight_ent = np.exp(-entropy)
        weight_loc = 1.0 - weight_ent
        w = np.array([weight_loc, weight_ent])
        w = w / w.sum()
        return self.engine.bundle([local, entangled], weights=w)

    def ascend(self, vectors: List[np.ndarray]) -> Tuple[np.ndarray, List[MERALevel]]:
        current = list(vectors)
        self.levels = []
        while len(current) > 1:
            level = MERALevel()
            nxt = []
            if len(current) % 2 == 1:
                current.append(current[-1].copy())
            for i in range(0, len(current), 2):
                a, b = current[i], current[i + 1]
                a_loc, b_ent, spec = self.disentangle_pair(a, b)
                comp = self.isometric_compress(a_loc, b_ent, spec)
                nxt.append(comp)
                level.disentanglers.append(spec)
            self.levels.append(level)
            current = nxt
        return current[0], self.levels

    def descend(self, top_vec: np.ndarray, n_leaves: int,
                levels: Optional[List[MERALevel]] = None) -> List[np.ndarray]:
        if levels is None:
            levels = self.levels
        current = [top_vec]
        for lvl in reversed(levels):
            nxt = []
            for comp in current:
                noise = np.random.normal(0, 0.05, self.engine.dim)
                left = (comp + noise) % (2 * np.pi)
                right = (comp - noise) % (2 * np.pi)
                nxt.extend([left, right])
            current = nxt[:n_leaves * 2]
        return current[:n_leaves]


# -----------------------------------------------------------------------------
# 5. Spectral Geometry & Diffusion Maps
# -----------------------------------------------------------------------------
class SpectralGeometry:
    def __init__(self, engine):
        self.engine = engine
        self.affinity: Optional[np.ndarray] = None
        self.laplacian: Optional[np.ndarray] = None
        self.laplacian_norm: Optional[np.ndarray] = None
        self.eigvals: Optional[np.ndarray] = None
        self.eigvecs: Optional[np.ndarray] = None

    def build_affinity(self, sigma: float = 0.15, k: int = 20) -> np.ndarray:
        n = len(self.engine.token_names)
        aff = np.zeros((n, n))
        for i in range(n):
            sims = np.array([
                self.engine.sim(self.engine.token_phases[i], self.engine.token_phases[j])
                for j in range(n)
            ])
            knn_thresh = np.partition(sims, -k)[-k] if n > k else -1.0
            for j in range(n):
                if sims[j] >= knn_thresh:
                    aff[i, j] = np.exp(-((1.0 - sims[j]) ** 2) / (2 * sigma ** 2))
        self.affinity = np.maximum(aff, aff.T)
        return self.affinity

    def compute_laplacian(self, normalization: str = 'symmetric'):
        if self.affinity is None:
            self.build_affinity()
        A = self.affinity
        D = np.diag(A.sum(axis=1))
        self.laplacian = D - A
        if normalization == 'symmetric':
            d_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-12))
            self.laplacian_norm = d_inv_sqrt @ self.laplacian @ d_inv_sqrt
        return self.laplacian_norm

    def spectral_embedding(self, n_components: int = 20) -> np.ndarray:
        if self.laplacian_norm is None:
            self.compute_laplacian()
        e, v = eigh(self.laplacian_norm)
        self.eigvals = e
        self.eigvecs = v
        return v[:, 1:n_components + 1]

    def diffusion_map(self, t: float = 2.0, n_components: int = 10) -> np.ndarray:
        if self.affinity is None:
            self.build_affinity()
        A = self.affinity
        D_inv = np.diag(1.0 / (A.sum(axis=1) + 1e-12))
        P = D_inv @ A
        e, v = np.linalg.eig(P)
        idx = np.argsort(-np.real(e))
        e = np.real(e[idx])
        v = np.real(v[:, idx])
        coords = np.zeros((len(e), n_components))
        for i in range(1, n_components + 1):
            coords[:, i - 1] = (e[i] ** t) * v[:, i]
        return coords

    def cheeger_clustering(self, n_clusters: int = 8) -> Dict[int, List[str]]:
        emb = self.spectral_embedding(n_components=n_clusters)
        centroids = emb[np.random.choice(len(emb), n_clusters, replace=False)]
        labels = np.zeros(len(emb), dtype=int)
        for _ in range(20):
            sims = emb @ centroids.T
            labels = np.argmax(sims, axis=1)
            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    centroids[k] = emb[mask].mean(axis=0)
        clusters = defaultdict(list)
        for idx, lab in enumerate(labels):
            clusters[int(lab)].append(self.engine.token_names[idx])
        return dict(clusters)


# -----------------------------------------------------------------------------
# 6. FHRR Topological Layer (Unified API)
# -----------------------------------------------------------------------------
class FHRRTopologicalLayer:
    def __init__(self, engine):
        self.engine = engine
        self.ph = VietorisRipsPH(max_dim=1, k_neighbors=12, max_dist=1.2)
        self.sheaf = SheafVSA(engine)
        self.bundle = FiberBundleVSA(engine)
        self.mera = MERAHierarchy(engine, block_size=64)
        self.spectral = SpectralGeometry(engine)
        self._dist_matrix: Optional[np.ndarray] = None
        self._cached_tokens: int = 0

    def _ensure_distance_matrix(self):
        n = len(self.engine.token_names)
        if self._dist_matrix is not None and self._cached_tokens == n:
            return self._dist_matrix
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = 1.0 - self.engine.sim(self.engine.token_phases[i], self.engine.token_phases[j])
                dist[i, j] = d
                dist[j, i] = d
        self._dist_matrix = dist
        self._cached_tokens = n
        return dist

    def analyze_vocabulary_topology(self) -> Dict[str, Any]:
        print("[Topo] Distance matrix...")
        D = self._ensure_distance_matrix()
        print("[Topo] Persistent Homology...")
        self.ph.fit(D)
        betti = self.ph.betti(threshold=0.5)
        print("[Topo] Sheaf & Fiber...")
        self.sheaf.build_stalks()
        self.sheaf.build_base_space()
        self.sheaf.compute_all_restrictions()
        self.bundle.build_fibers()
        self.bundle.build_connections(self.sheaf)
        h0 = self.sheaf.sheaf_cohomology_h0_estimate()
        print("[Topo] Spectral geometry...")
        emb = self.spectral.spectral_embedding(n_components=15)
        clusters = self.spectral.cheeger_clustering(n_clusters=10)
        return {
            'betti_numbers': betti,
            'significant_features': len(self.ph.significant_features(0.1)),
            'sheaf_h0': h0,
            'spectral_clusters': clusters,
            'embedding_shape': emb.shape
        }

    def mera_encode_sentence(self, bindings: Dict[str, str]) -> Tuple[np.ndarray, List[MERALevel]]:
        vectors = []
        for role_name, token_name in bindings.items():
            role_vec = self.engine.get_role(role_name)
            token_vec = self.engine.get_token(token_name)
            if role_vec is None or token_vec is None:
                continue
            bound = self.engine.bind(role_vec, token_vec, out=np.zeros(self.engine.dim))
            vectors.append(bound)
        if not vectors:
            raise ValueError("No valid bindings")
        return self.mera.ascend(vectors)

    def topological_query(self, query_vec: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        match, conf = self.engine.cleanup(query_vec)
        decoded = self.engine.decode(query_vec, threshold=0.35)
        cat_compat = {}
        for role, (tok, c) in decoded.items():
            cat = self.engine.token_categories[self.engine.token_names.index(tok)]
            if cat in self.sheaf.stalks:
                center = self.sheaf.stalks[cat].center
                cat_compat[role] = float(self.engine.sim(query_vec, center))
        kernel_results = self.engine.kernel_query(query_vec, radius=0.25, top_k=top_k)
        return {
            'cleanup_match': match,
            'cleanup_conf': conf,
            'decoded': decoded,
            'category_compatibility': cat_compat,
            'kernel_neighbors': kernel_results
        }

    def detect_topological_contradiction(self, vec1: np.ndarray, vec2: np.ndarray) -> Dict[str, Any]:
        is_contra, conflicts = self.engine.detect_contradiction(vec1, vec2)
        dec1 = self.engine.decode(vec1, threshold=0.35)
        dec2 = self.engine.decode(vec2, threshold=0.35)
        sheaf_conflicts = []
        for role in set(dec1.keys()) & set(dec2.keys()):
            t1, _ = dec1[role]
            t2, _ = dec2[role]
            c1 = self.engine.token_categories[self.engine.token_names.index(t1)]
            c2 = self.engine.token_categories[self.engine.token_names.index(t2)]
            if c1 != c2:
                sheaf_conflicts.append({
                    'role': role, 'token1': t1, 'cat1': c1,
                    'token2': t2, 'cat2': c2
                })
        curvature = None
        if len(self.bundle.connections) > 2:
            cats = list(self.bundle.fibers.keys())[:4]
            loop = cats + [cats[0]]
            curvature = self.bundle.curvature(loop, vec1)
        return {
            'standard_contradiction': is_contra,
            'standard_conflicts': conflicts,
            'sheaf_conflicts': sheaf_conflicts,
            'bundle_curvature': curvature
        }

    def analogy_via_fiber_transport(self, source_token: str, source_cat: str,
                                    target_cat: str) -> Tuple[Optional[str], float]:
        src_vec = self.engine.get_token(source_token)
        if src_vec is None:
            return None, 0.0
        path = self.bundle.compute_geodesic(source_cat, target_cat, self.sheaf)
        transported = self.bundle.parallel_transport(src_vec, path)
        best_tok = None
        best_sim = -1.0
        for idx, (name, cat) in enumerate(zip(self.engine.token_names, self.engine.token_categories)):
            if cat == target_cat:
                sim = self.engine.sim(transported, self.engine.token_phases[idx])
                if sim > best_sim:
                    best_sim = sim
                    best_tok = name
        return best_tok, float(best_sim)

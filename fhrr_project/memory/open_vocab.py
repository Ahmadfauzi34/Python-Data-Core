# =============================================================================
# MEMORY LAYER: Open Vocabulary & Compositional Morphology
# =============================================================================
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class OpenVocabularyExtension:
    def __init__(self, engine, novelty_threshold: float = 0.32,
                 min_confidence_for_register: float = 0.45):
        self.engine = engine
        self.novelty_threshold = novelty_threshold
        self.min_confidence = min_confidence_for_register
        self.composition_buffer = np.zeros(engine.dim)
        self._cbuf = np.zeros(engine.dim, dtype=np.complex128)

        self.neologisms: Dict[str, Dict] = {}
        self.category_prototypes: Dict[str, np.ndarray] = {}

        self._refresh_category_centers()

    def _refresh_category_centers(self):
        cats = defaultdict(list)
        for idx, cat in enumerate(self.engine.token_categories):
            cats[cat].append(self.engine.token_phases[idx])
        self.category_prototypes = {
            cat: self.engine.bundle(list(vecs)) if len(vecs) > 1 else vecs[0]
            for cat, vecs in cats.items()
        }

    def resolve_token(self, name: str, preferred_category: Optional[str] = None,
                      noise: float = 0.22) -> np.ndarray:
        existing = self.engine.get_token(name)
        if existing is not None:
            return existing

        if preferred_category and preferred_category in self.category_prototypes:
            proto = self.category_prototypes[preferred_category]
            cat = preferred_category
        else:
            proto = self.engine.alloc()
            cat = 'unknown'

        vec = self.engine.add_token(name, cat, prototype=proto, noise=noise)
        self._refresh_category_centers()
        return vec

    def compose_bind(self, head: str, modifier: str,
                     head_role: str = 'head', mod_role: str = 'modifier') -> Tuple[np.ndarray, str]:
        h_vec = self.resolve_token(head)
        m_vec = self.resolve_token(modifier)

        hr = self.engine.add_role(head_role)
        mr = self.engine.add_role(mod_role)

        bound_h = self.engine.bind(hr, h_vec, out=self.composition_buffer.copy())
        bound_m = self.engine.bind(mr, m_vec, out=self.composition_buffer.copy())

        composed = self.engine.bundle([bound_h, bound_m], out=self._cbuf.copy())

        auto_name = f"{head}_{modifier}"
        self.neologisms[auto_name] = {
            'head': head, 'modifier': modifier,
            'head_role': head_role, 'mod_role': mod_role,
            'vector': composed.copy(), 'inferred_cat': 'composite'
        }
        return composed, auto_name

    def compose_phrase(self, tokens: List[str], weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, str]:
        vecs = [self.resolve_token(t) for t in tokens]
        composed = self.engine.bundle(vecs, weights=weights, out=self._cbuf.copy())
        auto_name = "_".join(tokens)
        self.neologisms[auto_name] = {
            'components': tokens, 'vector': composed.copy(),
            'inferred_cat': 'phrase'
        }
        return composed, auto_name

    def try_cleanup_or_register(self, vec: np.ndarray, suggested_name: Optional[str] = None,
                                 suggested_category: Optional[str] = None) -> Tuple[str, float, bool]:
        match, sim = self.engine.cleanup(vec, threshold=self.novelty_threshold)
        if match is not None:
            return match, sim, False

        name = suggested_name or f"novel_{len(self.engine.token_names)}"
        inferred_cat = self._infer_category(vec) or suggested_category or 'unknown'
        self.engine.add_token(name, inferred_cat, prototype=vec, noise=0.05)
        self._refresh_category_centers()

        self.neologisms[name] = {
            'vector': vec.copy(), 'inferred_cat': inferred_cat,
            'auto_registered': True
        }
        return name, sim, True

    def _infer_category(self, vec: np.ndarray) -> Optional[str]:
        if not self.category_prototypes:
            return None
        best_cat = None
        best_sim = -1.0
        for cat, center in self.category_prototypes.items():
            sim = self.engine.sim(vec, center)
            if sim > best_sim:
                best_sim = sim
                best_cat = cat
        return best_cat if best_sim > 0.25 else None

    def decompose_neologism(self, name: str) -> Optional[Dict]:
        return self.neologisms.get(name)

    def is_composite(self, name: str) -> bool:
        return name in self.neologisms

def extend_engine_open_vocab(engine) -> OpenVocabularyExtension:
    """Helper untuk menempelkan open vocab extension ke engine yang sudah ada."""
    return OpenVocabularyExtension(engine)

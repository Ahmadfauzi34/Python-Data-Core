# =============================================================================
# AGENT LAYER: Recursive Self-Improvement Engine
# =============================================================================
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
from fhrr_project.memory.knowledge_graph import KGTriple

@dataclass
class AssessmentReport:
    timestamp: float
    coverage_score: float      
    coherence_score: float     
    entropy_score: float       
    gap_regions: List[Dict]    
    suggestions: List[Dict]    

class SelfImprovementEngine:
    def __init__(self, engine, topo_layer, discoverer, kg_ingestor):
        self.engine = engine
        self.topo = topo_layer          
        self.discoverer = discoverer    
        self.kg = kg_ingestor           
        self.assessments: List[AssessmentReport] = []
        self.improvement_log: List[Dict] = []
        self.min_coherence = 0.35
        self.min_coverage = 0.60
        self.max_gap_ratio = 0.25
        self._ws = np.zeros(engine.dim)

    def assess(self) -> AssessmentReport:
        t0 = time.time()
        n_tokens = len(self.engine.token_names)
        tokens_with_rel = set()
        for rel in self.discoverer.discovered_relations.values():
            tokens_with_rel.add(rel['from'])
            tokens_with_rel.add(rel['to'])
        coverage = len(tokens_with_rel) / max(1, n_tokens)

        if self.topo and len(self.topo.sheaf.stalks) > 0:
            assignment = {cat: stalk.center for cat, stalk in self.topo.sheaf.stalks.items()}
            is_consistent, violations = self.topo.sheaf.global_section_consistency(assignment, tol=0.35)
            coherence = 1.0 - (len(violations) / max(1, len(self.topo.sheaf.base_adj)))
        else:
            coherence = 0.5  

        role_counts = Counter()
        for r in self.discoverer.discovered_roles.values():
            role_counts[r['role']] += 1
        if role_counts:
            probs = np.array(list(role_counts.values())) / sum(role_counts.values())
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            max_ent = np.log(len(role_counts) + 1e-12)
            norm_entropy = entropy / max_ent if max_ent > 0 else 0
        else:
            norm_entropy = 0.0

        gaps = self._detect_gaps()
        suggestions = self._generate_suggestions(gaps)

        report = AssessmentReport(
            timestamp=t0, coverage_score=float(coverage), coherence_score=float(coherence),
            entropy_score=float(norm_entropy), gap_regions=gaps, suggestions=suggestions
        )
        self.assessments.append(report)
        return report

    def _detect_gaps(self) -> List[Dict]:
        gaps = []
        if not self.topo:
            return gaps

        if hasattr(self.topo.spectral, 'affinity') and self.topo.spectral.affinity is not None:
            emb = self.topo.spectral.spectral_embedding(n_components=10)
            for i, tok in enumerate(self.engine.token_names):
                dists = np.linalg.norm(emb - emb[i], axis=1)
                neighbors = np.sum(dists < 0.5)
                if neighbors < 3:
                    gaps.append({
                        'type': 'spectral_isolation', 'token': tok,
                        'category': self.engine.token_categories[i],
                        'neighbor_count': int(neighbors), 'severity': 1.0 - (neighbors / 3.0)
                    })

        if self.topo.sheaf.base_adj:
            for cat in self.topo.sheaf.stalks.keys():
                if cat not in self.topo.sheaf.base_adj or len(self.topo.sheaf.base_adj[cat]) == 0:
                    gaps.append({'type': 'sheaf_isolation', 'category': cat, 'severity': 0.8})

        if self.topo.ph.pairs:
            h1_features = [p for p in self.topo.ph.pairs if p.dim == 1]
            if len(h1_features) < 2:
                gaps.append({'type': 'ph_cycle_deficit', 'description': 'Kurang relasi siklik/kompleks', 'severity': 0.6})

        gaps.sort(key=lambda x: -x['severity'])
        return gaps

    def _generate_suggestions(self, gaps: List[Dict]) -> List[Dict]:
        suggestions = []

        # Pre-calculate stalk centers if sheaf_isolation gaps are present
        stalk_names = []
        stalk_centers = None
        if any(g['type'] == 'sheaf_isolation' for g in gaps[:5]) and self.topo.sheaf.stalks:
            stalk_names = list(self.topo.sheaf.stalks.keys())
            stalk_centers = np.stack([self.topo.sheaf.stalks[cat].center for cat in stalk_names])

        for gap in gaps[:5]:
            if gap['type'] == 'spectral_isolation':
                tok = gap['token']
                cat = gap['category']
                same_cat = [t for t, c in zip(self.engine.token_names, self.engine.token_categories) if c == cat and t != tok]
                if same_cat:
                    partner = np.random.choice(same_cat)
                    suggestions.append({
                        'target_gap': gap, 'action': 'generate_corpus',
                        'template': f"{tok} dan {partner} adalah dua jenis {cat}",
                        'rationale': f'Membuat asosiasi intra-kategori untuk {tok}'
                    })
            elif gap['type'] == 'sheaf_isolation':
                cat = gap['category']
                if stalk_centers is not None:
                    target_center = self.topo.sheaf.stalks[cat].center
                    # Vectorized similarity: mean(cos(A - B))
                    # A is target_center (dim,), B is stalk_centers (N, dim)
                    # result is (N,)
                    sims = np.mean(np.cos(stalk_centers - target_center), axis=1)

                    # Mask out the current category
                    cat_idx = stalk_names.index(cat)
                    sims[cat_idx] = -1.0

                    best_idx = np.argmax(sims)
                    best_sim = sims[best_idx]

                    if best_sim > -1:
                        best_cat = stalk_names[best_idx]
                        suggestions.append({
                            'target_gap': gap, 'action': 'generate_corpus',
                            'template': f"sesuatu yang {cat} sering digunakan dengan {best_cat}",
                            'rationale': f'Hubungkan {cat} ke {best_cat} via relasi fungsional'
                        })
            elif gap['type'] == 'ph_cycle_deficit':
                cats = list(self.topo.sheaf.stalks.keys())[:3]
                tokens = []
                for c in cats:
                    stalk = self.topo.sheaf.stalks.get(c)
                    if stalk and stalk.indices:
                        idx = np.random.choice(stalk.indices)
                        tokens.append(self.engine.token_names[idx])
                if len(tokens) >= 3:
                    suggestions.append({
                        'target_gap': gap, 'action': 'generate_corpus',
                        'template': f"{tokens[0]} menggunakan {tokens[1]} untuk membuat {tokens[2]}",
                        'rationale': 'Membuat rantai kausal 3-node untuk menambah siklus H1'
                    })
        return suggestions

    def improve_step(self, external_corpus_feed: Optional[List[str]] = None) -> Dict[str, Any]:
        before = self.assess()
        new_corpus = []
        if external_corpus_feed:
            new_corpus.extend(external_corpus_feed)
        else:
            for sug in before.suggestions:
                if sug['action'] == 'generate_corpus':
                    new_corpus.append(sug['template'])

        if new_corpus:
            self.discoverer.ingest_corpus(new_corpus)
            self.discoverer.mine_cooccurrence()
            self.discoverer.induce_relations()
            self.discoverer.discover_patterns()

            for sent in new_corpus:
                tokens = [t for t in sent.lower().split() if self.engine.get_token(t) is not None]
                if len(tokens) >= 3:
                    roles = [self.discoverer.discovered_roles.get(t, {}).get('role', '?') for t in tokens]
                    for i in range(len(tokens) - 2):
                        if roles[i].endswith('subject') and roles[i+1].endswith('predicate'):
                            self.kg.ingest_triple(KGTriple(
                                tokens[i], tokens[i+1], tokens[i+2],
                                {'source': 'self_improvement', 'auto': True}
                            ))

        after = self.assess()
        delta = {
            'coverage_delta': after.coverage_score - before.coverage_score,
            'coherence_delta': after.coherence_score - before.coherence_score,
            'entropy_delta': after.entropy_score - before.entropy_score
        }

        self.improvement_log.append({
            'iteration': len(self.improvement_log), 'before': before,
            'after': after, 'delta': delta, 'corpus_added': len(new_corpus)
        })

        return {'before': before, 'after': after, 'delta': delta, 'suggestions_executed': len(new_corpus)}

    def run_iterations(self, n: int = 3, external_feeds: Optional[List[List[str]]] = None) -> List[Dict]:
        results = []
        for i in range(n):
            feed = external_feeds[i] if external_feeds and i < len(external_feeds) else None
            result = self.improve_step(external_corpus_feed=feed)
            results.append(result)
        return results

    def get_improvement_trajectory(self) -> Dict[str, List[float]]:
        return {
            'coverage': [a.coverage_score for a in self.assessments],
            'coherence': [a.coherence_score for a in self.assessments],
            'entropy': [a.entropy_score for a in self.assessments],
            'timestamps': [a.timestamp for a in self.assessments]
        }

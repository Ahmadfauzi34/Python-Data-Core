import numpy as np
import time
import yaml
import os
import logging

logger = logging.getLogger(__name__)
from typing import List, Dict, Any, Tuple
from fhrr_project.core.engine import FHRREngine
from fhrr_project.core.roles import Role

class MetaCognitiveConsolidator:
    """
    Mewakili fase "Tidur" atau "Refleksi" (Sleep Phase).
    Agen menganalisis Episodic Buffer-nya (ingatan jangka pendek) untuk
    menemukan transformasi vektor yang berulang (Pattern Induction).
    Jika pola yang valid ditemukan, agen akan mengekstraknya sebagai aturan logis
    baru dan menulisnya secara permanen ke sistem.
    """
    def __init__(self, engine: FHRREngine, dataset_dir: str):
        self.engine = engine
        self.dataset_dir = dataset_dir
        self.min_cluster_size = 2
        self.similarity_threshold_explicit = 0.85
        self.similarity_threshold_temporal = 0.35

    def extract_transformations(self) -> List[Dict]:
        """
        Melakukan FHRR Unbinding (Arah Vektor / Phase Difference) antara
        dua keadaan untuk mencari relasi sebab-akibat atau transformasi.
        Contoh: "Budi makan -> Budi kenyang".
        Vektor Transformasi T = Kenyang ⊘ Makan.
        """
        transformations = []

        # 1. Iterating over the history of explicitly learned one-off transforms
        if hasattr(self.engine, 'learned_transforms_history'):
            for record in self.engine.learned_transforms_history:
                transformations.append({
                    'from': record['from'],
                    'to': record['to'],
                    'vector': self.engine.transforms[record['name']]['vector'],
                    'source': 'explicit'
                })

        # 2. Temporal Episodic Causation (Unsupervised Sequence Learning)
        # Scan the episodic buffer, sort chronologically, and extract phase diffs between t and t+1
        if hasattr(self.engine, 'episodic_buffer') and len(self.engine.episodic_buffer) > 1:
            # Sort episodes by timestamp
            episodes = sorted(self.engine.episodic_buffer, key=lambda x: x.get('timestamp', 0))

            for i in range(len(episodes) - 1):
                ep1 = episodes[i]
                ep2 = episodes[i+1]

                # Check if they happened relatively close to each other (e.g. same document ingest session)
                time_diff = ep2.get('timestamp', 0) - ep1.get('timestamp', 0)
                # If they are within 60 seconds of each other, assume potential causation
                if 0 <= time_diff <= 60.0:
                    meta1 = ep1.get('metadata', {})
                    meta2 = ep2.get('metadata', {})
                    bind1 = meta1.get('bindings', {})
                    bind2 = meta2.get('bindings', {})

                    if bind1 and bind2:
                        # Strict Predicate-Causation Focus Extraction
                        # We only induce temporal causation if we can clearly identify a predicate-to-predicate
                        # (or predicate-to-attribute) sequence. Prevents asymmetric noise ("turun" -> "tanah").
                        t1_focus = bind1.get('predikat')
                        t2_focus = bind2.get('predikat') or bind2.get('atribut')

                        if t1_focus and t2_focus and t1_focus != t2_focus:
                            # FHRR Phase Difference: T_causal = Event_{t+1} ⊘ Event_t
                            v1 = ep1['vector']
                            v2 = ep2['vector']
                            diff_vec = (v2 - v1 + np.pi) % (2 * np.pi) - np.pi

                            transformations.append({
                                'from': t1_focus,
                                'to': t2_focus,
                                'vector': diff_vec,
                                'source': 'temporal'
                            })

        return transformations

    def _cluster_pool(self, pool: List[Dict], threshold: float, mechanism_tag: str) -> List[Dict]:
        if not pool:
            return []

        n = len(pool)
        vectors = np.stack([t['vector'] for t in pool])

        C = np.cos(vectors)
        S = np.sin(vectors)
        sim_mat = (C @ C.T + S @ S.T) / self.engine.dim

        visited = set()
        induced_rules = []

        for i in range(n):
            if i in visited:
                continue

            cluster_indices = [i]
            for j in range(i + 1, n):
                if j not in visited and sim_mat[i, j] >= threshold:
                    cluster_indices.append(j)

            if len(cluster_indices) >= self.min_cluster_size:
                visited.update(cluster_indices)

                evidence = [pool[idx] for idx in cluster_indices]
                from_tokens = [e['from'] for e in evidence]
                to_tokens = [e['to'] for e in evidence]

                from_majority = max(set(from_tokens), key=from_tokens.count)
                to_majority = max(set(to_tokens), key=to_tokens.count)

                rule_name = f"auto_induced_{from_majority}_{to_majority}"

                other_indices = [j for j in cluster_indices if j != i]
                if other_indices:
                    conf = float(round(np.mean([sim_mat[i, j] for j in other_indices]), 3))
                else:
                    conf = 1.0

                semantic_signature = f"premise:predikat={from_majority}|conclusion:atribut={to_majority}"

                new_rule = {
                    'id': f"auto_r_{int(time.time())}_{i}",
                    'name': rule_name,
                    'semantic_signature': semantic_signature,
                    'premise': {'predikat': from_majority},
                    'conclusion': {'atribut': to_majority},
                    'mechanism': mechanism_tag,
                    'confidence': conf,
                    'explanation': f"Auto-induced from {len(evidence)} {mechanism_tag} similarities."
                }

                induced_rules.append(new_rule)

                # add_rule requires valid roles to exist in the engine.
                for role in new_rule['premise'].keys():
                    if role not in self.engine.role_names:
                        self.engine.add_role(role)

                self.engine.add_rule(
                    pattern_bindings=new_rule['premise'],
                    action=new_rule['conclusion'].get('atribut', ''),
                    confidence=new_rule['confidence'],
                    metadata={"source": "meta_consolidation"}
                )

        return induced_rules

    def consolidate(self) -> List[Dict]:
        """
        Mengelompokkan vektor-vektor transformasi dengan memisahkan pool
        sumber eksplisit dan sumber kausalitas temporal untuk menghindari noise.
        """
        transforms = self.extract_transformations()
        if not transforms:
            return []

        # Split pools
        pool_explicit = [t for t in transforms if t.get('source') == 'explicit']
        pool_temporal = [t for t in transforms if t.get('source') == 'temporal']

        induced_rules = []
        induced_rules.extend(self._cluster_pool(pool_explicit, self.similarity_threshold_explicit, 'transform'))
        induced_rules.extend(self._cluster_pool(pool_temporal, self.similarity_threshold_temporal, 'temporal_causation'))

        return induced_rules



    def persist_rules_to_dataset(self, new_rules: List[Dict]):
        """
        Meta-Learning: AI menulis kembali aturan yang ia temukan secara mandiri
        ke dalam persistent dataset miliknya (`reasoning_patterns.yaml`).
        """
        if not new_rules:
            return

        # Safety Guardrail: Write to a separate staging file
        filepath = os.path.join(self.dataset_dir, "reasoning_patterns.auto.yaml")

        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            data = {'reasoning_patterns': []}

        if 'reasoning_patterns' not in data:
            data['reasoning_patterns'] = []

        # Avoid duplicates based on semantic signature rather than generated name
        existing_signatures = {r.get('semantic_signature') for r in data['reasoning_patterns'] if r.get('semantic_signature')}

        added_count = 0
        for rule in new_rules:
            sig = rule.get('semantic_signature')
            if sig and sig not in existing_signatures:
                data['reasoning_patterns'].append(rule)
                existing_signatures.add(sig)
                added_count += 1

        if added_count > 0:
            with open(filepath, "w") as f:
                yaml.dump(data, f, sort_keys=False, allow_unicode=True)
            logger.info(f"Permanently saved {added_count} new meta-rules to {filepath}")

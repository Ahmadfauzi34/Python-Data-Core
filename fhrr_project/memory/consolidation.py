import numpy as np
import time
import yaml
import os
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
    def __init__(self, engine: FHRREngine, dataset_dir: str = "fhrr_project/data/datasets/default/"):
        self.engine = engine
        self.dataset_dir = dataset_dir
        self.min_cluster_size = 2
        self.similarity_threshold = 0.85

    def extract_transformations(self) -> List[Dict]:
        """
        Melakukan FHRR Unbinding (Arah Vektor / Phase Difference) antara
        dua keadaan untuk mencari relasi sebab-akibat atau transformasi.
        Contoh: "Budi makan -> Budi kenyang".
        Vektor Transformasi T = Kenyang ⊘ Makan.
        """
        # Simplify: We look at the engine's learned transforms history or episodic buffer pairs.
        # For an advanced implementation, we scan the engine.learned_transforms_history
        # or we actively extract diffs from sequential episodic memories.

        transformations = []

        # Iterating over the history of explicitly learned one-off transforms
        if hasattr(self.engine, 'learned_transforms_history'):
            for record in self.engine.learned_transforms_history:
                transformations.append({
                    'from': record['from'],
                    'to': record['to'],
                    'vector': self.engine.transforms[record['name']]['vector']
                })

        return transformations

    def consolidate(self) -> List[Dict]:
        """
        Mengelompokkan vektor-vektor transformasi. Jika banyak transformasi
        memiliki arah vektor (phase diff) yang mirip, ini mengindikasikan
        sebuah "Hukum Semantik" (Semantic Law) yang konsisten.
        """
        transforms = self.extract_transformations()
        if not transforms:
            return []

        n = len(transforms)
        vectors = np.stack([t['vector'] for t in transforms])

        # O(N^2) Vectorized clustering (using trig identity for FHRR sim)
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
                if j not in visited and sim_mat[i, j] >= self.similarity_threshold:
                    cluster_indices.append(j)

            if len(cluster_indices) >= self.min_cluster_size:
                visited.update(cluster_indices)

                # Ekstrak rule
                evidence = [transforms[idx] for idx in cluster_indices]

                # Coba cari common pattern
                # Heuristik: "Jika memakan -> kenyang", "Jika meminum -> kenyang" dsb.
                # Kita akan menamai rule ini berdasarkan aksi mayoritas.
                from_tokens = [e['from'] for e in evidence]
                to_tokens = [e['to'] for e in evidence]

                rule_name = f"auto_induced_{from_tokens[0]}_{to_tokens[0]}"

                new_rule = {
                    'id': f"auto_r_{int(time.time())}_{i}",
                    'name': rule_name,
                    'premise': {'predikat': from_tokens[0]}, # Simplified heuristic
                    'conclusion': {'atribut': to_tokens[0]}, # Simplified heuristic
                    'mechanism': 'transform',
                    'confidence': float(round(np.mean([sim_mat[i, j] for j in cluster_indices]), 3)),
                    'explanation': f"Auto-induced from {len(evidence)} episodic similarities."
                }

                induced_rules.append(new_rule)

                # Daftarkan ke memori aktif mesin
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

    def persist_rules_to_dataset(self, new_rules: List[Dict]):
        """
        Meta-Learning: AI menulis kembali aturan yang ia temukan secara mandiri
        ke dalam persistent dataset miliknya (`reasoning_patterns.yaml`).
        """
        if not new_rules:
            return

        filepath = os.path.join(self.dataset_dir, "reasoning_patterns.yaml")

        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            data = {'reasoning_patterns': []}

        if 'reasoning_patterns' not in data:
            data['reasoning_patterns'] = []

        # Avoid duplicates based on name
        existing_names = {r.get('name') for r in data['reasoning_patterns']}

        added_count = 0
        for rule in new_rules:
            if rule['name'] not in existing_names:
                data['reasoning_patterns'].append(rule)
                existing_names.add(rule['name'])
                added_count += 1

        if added_count > 0:
            with open(filepath, "w") as f:
                yaml.dump(data, f, sort_keys=False, allow_unicode=True)
            print(f"[Consolidator] Permanently saved {added_count} new meta-rules to {filepath}")

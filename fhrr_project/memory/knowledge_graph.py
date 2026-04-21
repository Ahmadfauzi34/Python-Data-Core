# =============================================================================
# MEMORY LAYER: Knowledge Graph Ingestor
# =============================================================================
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

@dataclass
class KGTriple:
    subject: str
    predicate: str
    object: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class KnowledgeGraphIngestor:
    def __init__(self, engine, open_vocab_extension):
        self.engine = engine
        self.open_vocab = open_vocab_extension
        self.triples_encoded: List[Dict[str, Any]] = [] 
        self.triples_vec: List[np.ndarray] = []

        self.s_role = self.engine.get_role('subject_role') or self.engine.add_role('subject_role')
        self.p_role = self.engine.get_role('predicate_role') or self.engine.add_role('predicate_role')
        self.o_role = self.engine.get_role('object_role') or self.engine.add_role('object_role')

    def _encode_triple_elements(self, s: str, p: str, o: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s_vec = self.open_vocab.resolve_token(s, preferred_category='entity')
        p_vec = self.open_vocab.resolve_token(p, preferred_category='relation')
        o_vec = self.open_vocab.resolve_token(o, preferred_category='entity')
        return s_vec, p_vec, o_vec

    def ingest_triple(self, triple: KGTriple):
        s_vec, p_vec, o_vec = self._encode_triple_elements(triple.subject, triple.predicate, triple.object)

        s_bound = self.engine.bind(self.s_role, s_vec)
        p_bound = self.engine.bind(self.p_role, p_vec)
        o_bound = self.engine.bind(self.o_role, o_vec)

        triple_vec = self.engine.bundle([s_bound, p_bound, o_bound])
        self.triples_encoded.append({'subject': triple.subject, 'predicate': triple.predicate, 'object': triple.object, 'metadata': triple.metadata})
        self.triples_vec.append(triple_vec)
        self.engine.store_episodic(triple_vec, metadata={'type': 'kg_triple', **triple.metadata})

    def ingest_batch(self, triples: List[KGTriple]):
        for triple in triples:
            self.ingest_triple(triple)

    def query_entity(self, entity: str, top_k: int = 3) -> List[Dict[str, Any]]:
        entity_vec = self.open_vocab.resolve_token(entity)

        results = []
        for i, t_vec in enumerate(self.triples_vec):
            sim_s = self.engine.sim(self.engine.unbind(t_vec, self.s_role), entity_vec)
            sim_o = self.engine.sim(self.engine.unbind(t_vec, self.o_role), entity_vec)

            if sim_s > 0.4 or sim_o > 0.4:
                results.append({
                    'triple': self.triples_encoded[i],
                    'similarity': max(sim_s, sim_o)
                })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def transitive_query(self, start_entity: str, relation_name: str, hops: int = 1) -> List[Tuple[str, float]]:
        path = [(start_entity, 1.0)]
        current_entity = start_entity
        current_vec = self.open_vocab.resolve_token(start_entity)

        for _ in range(hops):
            found_next = False
            for t_idx, triple_data in enumerate(self.triples_encoded):
                if triple_data['subject'] == current_entity and triple_data['predicate'] == relation_name:
                    next_entity = triple_data['object']
                    path.append((next_entity, self.engine.sim(current_vec, self.open_vocab.resolve_token(next_entity))))
                    current_entity = next_entity
                    current_vec = self.open_vocab.resolve_token(next_entity)
                    found_next = True
                    break
            if not found_next:
                break
        return path

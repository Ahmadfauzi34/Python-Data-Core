# =============================================================================
# INTERFACE LAYER: Natural Language Query Handler
# =============================================================================
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
from collections import Counter

from fhrr_project.core.roles import Role, TripleKey
from fhrr_project.interface.text_normalizer import (
    normalize, find_question_prefix, find_mentions, build_variant_index,
)

@dataclass
class QueryResult:
    query: str
    answer: str
    confidence: float
    mechanism: str
    reasoning: str
    related_entities: List[Dict]
    explanation: Optional[str] = None
    suggested_followup: Optional[List[str]] = None

class FHRRQueryInterface:
    def __init__(self, runner):
        self.runner = runner
        self.engine = runner.engine
        self.kg = None
        self.discoverer = None
        self.improver = None

        self.qpatterns = {
            'who': {'prefixes': ['siapa', 'who', 'siapakah', 'orang mana'], 'target_role': Role.AGEN, 'strategy': 'find_subject'},
            'what': {'prefixes': ['apa', 'what', 'apakah', 'hal apa'], 'target_role': Role.PASIEN, 'strategy': 'find_object'},
            'where': {'prefixes': ['di mana', 'where', 'dimanakah', 'tempat mana'], 'target_role': Role.LOKASI, 'strategy': 'find_location'},
            'when': {'prefixes': ['kapan', 'when', 'waktu kapan'], 'target_role': Role.WAKTU, 'strategy': 'find_time'},
            'why': {'prefixes': ['mengapa', 'kenapa', 'why'], 'target_role': Role.SUMBER, 'strategy': 'find_cause'},
            'how': {'prefixes': ['bagaimana', 'how', 'seperti apa'], 'target_role': Role.MANNER, 'strategy': 'find_manner'},
            'is_true': {'prefixes': ['apakah', 'is', 'does', 'benarkah'], 'target_role': 'verify', 'strategy': 'verify_fact'},
            'analogy': {'prefixes': ['seperti apa', 'analogi', 'ibarat', 'serupa dengan'], 'target_role': 'analogy', 'strategy': 'fiber_transport'}
        }

        self.query_history: List[Dict] = []
        self.feedback_buffer: List[Dict] = []

    def attach_kg(self, kg_ingestor):
        self.kg = kg_ingestor
        return self

    def attach_discoverer(self, discoverer):
        self.discoverer = discoverer
        return self

    def attach_improver(self, improver):
        self.improver = improver
        return self

    def _get_variant_index(self) -> dict[str, str]:
        """Cache index varian->stem. Re-build kalau vocab berubah."""
        cached = getattr(self, "_variant_index_cache", None)
        cached_size = getattr(self, "_variant_index_size", -1)
        if cached is None or cached_size != len(self.engine.token_names):
            cached = build_variant_index(self.engine.token_names)
            self._variant_index_cache = cached
            self._variant_index_size = len(self.engine.token_names)
        return cached

    def parse_query(self, query: str) -> Dict[str, Any]:
        # 1) Tentukan tipe pertanyaan via prefix tanya (longest-match, normalized).
        all_prefixes: list[tuple[str, str]] = [
            (prefix, qkey) for qkey, qinfo in self.qpatterns.items() for prefix in qinfo['prefixes']
        ]
        # Sort longest-first agar "siapakah" menang dari "siapa".
        all_prefixes.sort(key=lambda x: -len(x[0]))
        matched = find_question_prefix(query, [p for p, _ in all_prefixes])

        qtype, strategy, target_role = 'unknown', 'fallback', None
        if matched is not None:
            qkey = next(k for p, k in all_prefixes if p == matched)
            qtype = qkey
            strategy = self.qpatterns[qkey]['strategy']
            target_role = self.qpatterns[qkey]['target_role']

        # 2) Ekstrak entity & predikat dengan word-boundary + varian verba.
        idx = self._get_variant_index()
        mentioned = find_mentions(query, idx)
        predicates = [
            tok for tok in mentioned
            if 'aksi' in self.engine.token_categories[self.engine._token_name_to_idx[tok]]
        ]

        return {
            'raw': query, 'normalized': normalize(query),
            'qtype': qtype, 'strategy': strategy,
            'target_role': target_role, 'mentioned_entities': mentioned,
            'mentioned_predicates': predicates, 'parsed_at': time.time()
        }

    def ask(self, query: str, explain: bool = False) -> QueryResult:
        parsed = self.parse_query(query)
        self.query_history.append(parsed)

        handler = getattr(self, f'_handle_{parsed["strategy"]}', self._handle_fallback)
        result = handler(parsed, query)

        if explain:
            result.explanation = self._generate_explanation(result, parsed)
        result.suggested_followup = self._suggest_followup(result, parsed)

        return result

    def _handle_find_subject(self, parsed: Dict, raw: str) -> QueryResult:
        entities = parsed['mentioned_entities']
        predicates = parsed['mentioned_predicates']
        bindings = {}

        for ent in entities:
            cat = self.engine.token_categories[self.engine._token_name_to_idx[ent]]
            bindings[Role.PASIEN] = ent
        for pred in predicates:
            bindings[Role.PREDIKAT] = pred

        if not bindings:
            return QueryResult(raw, "Tidak dapat memahami query", 0.0, 'failed', 'Tidak ada entity dikenali', [])

        q_vec = self.engine.encode(bindings)
        if q_vec is None:
            return QueryResult(raw, "Encoding gagal", 0.0, 'failed', 'Tidak bisa encode bindings', [])

        # 1. Semantic Reasoning (Vector Unbind)
        role_vec = self.engine.get_role(Role.AGEN)
        if role_vec is not None:
            unbound = self.engine.unbind(q_vec, role_vec, out=np.zeros(self.engine.dim))
            match, sim = self.engine.cleanup(unbound, threshold=0.45)
            if match:
                return QueryResult(raw, match, sim, 'vector_inference', 'Diekstrak via unbinding semantic FHRR', [])

        # 2. Episodic Memory (Recent chat history)
        epi_match, epi_sim = self.engine.query_episodic(q_vec, threshold=0.40)
        if epi_match:
            # Assuming episodic metadata contains the answer or we re-cleanup the vector
            # For simplicity, let's unbind the agen from the episodic vector
            if role_vec is not None:
                unbound_epi = self.engine.unbind(epi_match['vector'], role_vec, out=np.zeros(self.engine.dim))
                match_epi, sim_epi = self.engine.cleanup(unbound_epi, threshold=0.40)
                if match_epi:
                    return QueryResult(raw, match_epi, epi_sim, 'episodic_recall', 'Diingat dari konteks percakapan (Episodic Memory)', [])

        # 3. Knowledge Graph Fallback (Rote Facts)
        best_match = None
        best_sim = -1.0

        if self.kg:
            for ent in entities:
                related = self.kg.query_entity(ent, top_k=5)
                for r in related:
                    triple = r['triple']
                    if triple.get(TripleKey.PREDICATE) in predicates:
                        sim = r['similarity']
                        if sim > best_sim:
                            best_sim = sim
                            best_match = triple

        if best_match:
            answer = best_match.get(TripleKey.SUBJECT, 'tidak diketahui')
            return QueryResult(raw, answer, best_sim, 'kg_lookup', f"Ditemukan di KG: {best_match.get(TripleKey.SUBJECT)} {best_match.get(TripleKey.PREDICATE)} {best_match.get(TripleKey.OBJECT)}", [{'entity': best_match.get(TripleKey.OBJECT), 'relation': best_match.get(TripleKey.PREDICATE), 'sim': best_sim}])

        return QueryResult(raw, "Tidak ditemukan", 0.0, 'not_found', 'Tidak ada match di Memori, Vector, maupun KG', [])

    def _handle_find_object(self, parsed: Dict, raw: str) -> QueryResult:
        entities = parsed['mentioned_entities']
        predicates = parsed['mentioned_predicates']
        bindings = {}

        for ent in entities:
            cat = self.engine.token_categories[self.engine._token_name_to_idx[ent]]
            if 'manusia' in cat or 'hewan' in cat:
                bindings[Role.AGEN] = ent
        for pred in predicates:
            bindings[Role.PREDIKAT] = pred

        q_vec = self.engine.encode(bindings)
        if q_vec is None: return QueryResult(raw, "Encoding gagal", 0.0, 'failed', '', [])

        # 1. Semantic Reasoning (Vector Unbind)
        role_vec = self.engine.get_role(Role.PASIEN)
        if role_vec is not None:
            unbound = self.engine.unbind(q_vec, role_vec, out=np.zeros(self.engine.dim))
            match, sim = self.engine.cleanup(unbound, threshold=0.45)
            if match:
                return QueryResult(raw, match, sim, 'vector_inference', 'Diekstrak via unbinding semantic FHRR', [])

        # 2. Episodic Memory (Recent chat history)
        epi_match, epi_sim = self.engine.query_episodic(q_vec, threshold=0.40)
        if epi_match:
            if role_vec is not None:
                unbound_epi = self.engine.unbind(epi_match['vector'], role_vec, out=np.zeros(self.engine.dim))
                match_epi, sim_epi = self.engine.cleanup(unbound_epi, threshold=0.40)
                if match_epi:
                    return QueryResult(raw, match_epi, epi_sim, 'episodic_recall', 'Diingat dari konteks percakapan (Episodic Memory)', [])

        # 3. Knowledge Graph Fallback (Rote Facts)
        if self.kg:
            for ent in entities:
                related = self.kg.query_entity(ent, top_k=5)
                for r in related:
                    triple = r['triple']
                    if triple.get(TripleKey.SUBJECT) == ent and triple.get(TripleKey.PREDICATE) in predicates:
                        return QueryResult(raw, triple.get(TripleKey.OBJECT, 'tidak diketahui'), r['similarity'], 'kg_lookup', f"Ditemukan di KG: {triple[TripleKey.SUBJECT]} {triple[TripleKey.PREDICATE]} {triple[TripleKey.OBJECT]}", [])

        return QueryResult(raw, "Tidak ditemukan", 0.0, 'not_found', 'Tidak ada match di Memori, Vector, maupun KG', [])

    def _handle_find_location(self, parsed: Dict, raw: str) -> QueryResult:
        return self._handle_role_query(parsed, raw, Role.LOKASI)

    def _handle_find_time(self, parsed: Dict, raw: str) -> QueryResult:
        return self._handle_role_query(parsed, raw, Role.WAKTU)

    def _handle_find_cause(self, parsed: Dict, raw: str) -> QueryResult:
        entities = parsed['mentioned_entities']
        for ent in entities:
            for tname, tinfo in self.engine.transforms.items():
                if tinfo.get('target') == ent and 'causal' in tname:
                    src = tinfo.get('source')
                    return QueryResult(raw, src, tinfo['confidence'], 'causal_transform', f"Inferensi kausal: {src} menyebabkan {ent}", [{'cause': src, 'effect': ent, 'transform': tname}])
        return QueryResult(raw, "Penyebab tidak diketahui", 0.0, 'no_causal_link', 'Tidak ada transform kausal yang cocok', [])

    def _handle_verify_fact(self, parsed: Dict, raw: str) -> QueryResult:
        entities = parsed['mentioned_entities']
        predicates = parsed['mentioned_predicates']
        bindings = {}
        for ent in entities:
            cat = self.engine.token_categories[self.engine._token_name_to_idx[ent]]
            if 'manusia' in cat: bindings[Role.AGEN] = ent
            elif 'aksi' in cat: bindings[Role.PREDIKAT] = ent
            else: bindings[Role.PASIEN] = ent

        q_vec = self.engine.encode(bindings)
        if q_vec is None: return QueryResult(raw, "tidak tahu", 0.0, 'failed', '', [])

        best_sim, best_entry = -1.0, None
        for entry in self.engine.episodic_buffer:
            sim = self.engine.sim(q_vec, entry['vector'])
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        threshold = 0.55
        if best_entry and best_sim > threshold:
            return QueryResult(raw, "ya", best_sim, 'fact_verified', f"Fakta cocok dengan memory (sim={best_sim:.3f})", [{'match': best_entry.get('metadata', {})}])
        else:
            if best_entry:
                is_contra, conflicts = self.engine.detect_contradiction(q_vec, best_entry['vector'])
                if is_contra:
                    return QueryResult(raw, "tidak", best_sim, 'contradiction_detected', f"Bertentangan dengan fakta yang ada: {conflicts}", [])
            return QueryResult(raw, "tidak tahu", best_sim if best_entry else 0.0, 'insufficient_data', 'Tidak ada fakta yang cukup mirip', [])

    def _handle_fiber_transport(self, parsed: Dict, raw: str) -> QueryResult:
        entities = parsed['mentioned_entities']
        if len(entities) < 2: return QueryResult(raw, "Butuh 2 entity untuk analogi", 0.0, 'failed', '', [])

        src = entities[0]
        tgt_cat = None
        if len(entities) > 1:
            tgt = entities[1]
            tgt_cat = self.engine.token_categories[self.engine._token_name_to_idx[tgt]]

        if not tgt_cat: return QueryResult(raw, "Kategori target tidak jelas", 0.0, 'failed', '', [])
        src_cat = self.engine.token_categories[self.engine._token_name_to_idx[src]]

        if self.runner.topo:
            result, sim = self.runner.topo.analogy_via_fiber_transport(src, src_cat, tgt_cat)
            if result:
                return QueryResult(raw, result, sim, 'fiber_transport', f"Analogi: {src} ({src_cat}) -> {result} ({tgt_cat}) via bundle transport", [{'source': src, 'target': result, 'similarity': sim}])
        return QueryResult(raw, "Tidak dapat membuat analogi", 0.0, 'no_analogy', '', [])

    def _handle_fallback(self, parsed: Dict, raw: str) -> QueryResult:
        entities = parsed['mentioned_entities']
        if not entities: return QueryResult(raw, "Tidak mengerti pertanyaan", 0.0, 'unparseable', 'Tidak ada keyword yang dikenali', [])

        vecs = [self.engine.get_token(e) for e in entities if self.engine.get_token(e) is not None]
        if vecs:
            avg_vec = self.engine.bundle(vecs)
            neighbors = self.engine.kernel_query(avg_vec, radius=0.3, top_k=3)
            suggestions = [n[0] for n in neighbors]
            return QueryResult(
                raw, "Tidak pasti", max(n[1] for n in neighbors) if neighbors else 0,
                'similarity_fallback', 'Mungkin yang Anda maksud: ' + ', '.join(suggestions),
                [{'token': n[0], 'sim': n[1]} for n in neighbors],
                suggested_followup=[f"Apakah Anda maksud {s}?" for s in suggestions[:2]]
            )
        return QueryResult(raw, "Tidak mengerti", 0.0, 'unparseable', '', [])

    def _handle_role_query(self, parsed: Dict, raw: str, role: str) -> QueryResult:
        entities = parsed['mentioned_entities']
        predicates = parsed['mentioned_predicates']
        bindings = {}
        for ent in entities:
            cat = self.engine.token_categories[self.engine._token_name_to_idx[ent]]
            if 'manusia' in cat or 'hewan' in cat: bindings[Role.AGEN] = ent
        for pred in predicates: bindings[Role.PREDIKAT] = pred

        q_vec = self.engine.encode(bindings)
        if q_vec is None: return QueryResult(raw, "Encoding gagal", 0.0, 'failed', '', [])

        role_vec = self.engine.get_role(role)
        if role_vec is not None and q_vec is not None:
            unbound = self.engine.unbind(q_vec, role_vec, out=np.zeros(self.engine.dim))
            match, sim = self.engine.cleanup(unbound)
            if match:
                return QueryResult(raw, match, sim, 'role_unbind', f"Diekstrak via unbinding role {role}", [])
        return QueryResult(raw, "Tidak ditemukan", 0.0, 'not_found', '', [])

    def _generate_explanation(self, result: QueryResult, parsed: Dict) -> str:
        explanations = [
            f"Pertanyaan: '{result.query}'", f"Jawaban: {result.answer}",
            f"Keyakinan: {result.confidence:.1%}", f"Mekanisme: {result.mechanism}"
        ]
        if result.reasoning: explanations.append(f"Alasan: {result.reasoning}")
        if result.related_entities:
            explanations.append("Entitas terkait:")
            for e in result.related_entities: explanations.append(f"  - {e}")
        return "\n".join(explanations)

    def _suggest_followup(self, result: QueryResult, parsed: Dict) -> List[str]:
        suggestions = []
        qtype = parsed['qtype']
        if qtype == 'who':
            suggestions.extend([f"Apa yang dilakukan {result.answer}?", f"Di mana {result.answer} berada?"])
        elif qtype == 'what':
            suggestions.extend(["Siapa yang melakukan itu?", "Bagaimana prosesnya?"])
        elif qtype == 'where':
            suggestions.append(f"Kapan kejadian di {result.answer}?")
        elif qtype == 'why':
            suggestions.append(f"Apa akibat dari {result.answer}?")
        elif result.mechanism == 'not_found':
            suggestions.extend(["Bisakah Anda memberikan informasi tambahan?", "Apakah ada konteks lain?"])
        return suggestions[:2]

    def provide_feedback(self, query: str, correct_answer: str, user_rating: float = 1.0) -> bool:
        self.feedback_buffer.append({'query': query, 'expected': correct_answer, 'rating': user_rating, 'timestamp': time.time()})
        if user_rating < 0.5 and self.improver:
            suggestion = f"{query} jawabannya adalah {correct_answer}"
            self.improver.improve_step(external_corpus_feed=[suggestion])
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_queries': len(self.query_history),
            'avg_confidence': np.mean([q.get('confidence', 0) for q in self.query_history]) if self.query_history else 0,
            'feedback_count': len(self.feedback_buffer),
            'qtype_distribution': Counter(q['qtype'] for q in self.query_history)
        }

import re
from typing import List, Dict, Any, Optional
from fhrr_project.core.engine import FHRREngine
from fhrr_project.core.roles import Role

# Optional import for KG integration
try:
    from fhrr_project.memory.knowledge_graph import KnowledgeGraphIngestor, KGTriple
except ImportError:
    KnowledgeGraphIngestor = None
    KGTriple = None


class TextIngestorBlueprint:
    """
    Blueprint untuk pipeline NLP Unstructured Text to FHRR.
    Mengubah paragraf bebas menjadi FHRR phase vectors.
    """
    def __init__(self, engine: FHRREngine, kg_ingestor: Optional[Any] = None):
        self.engine = engine
        self.kg = kg_ingestor

        # Preposition mapping to Roles
        self.prep_role_map = {
            'di': Role.LOKASI,
            'ke': Role.LOKASI,
            'dari': Role.LOKASI,
            'pada': Role.WAKTU,
            'saat': Role.WAKTU,
            'ketika': Role.WAKTU,
            'dengan': Role.INSTRUMEN,
            'pakai': Role.INSTRUMEN,
            'menggunakan': Role.INSTRUMEN,
            'oleh': Role.AGEN,
            'untuk': Role.PASIEN
        }

    def _chunk_sentences(self, text: str) -> List[str]:
        """Memecah paragraf menjadi list kalimat."""
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _extract_entities_and_roles(self, sentence: str) -> Dict[str, str]:
        """
        Rule-based NLP Heuristics for Indonesian.
        """
        # Tokenize preserving underscores for compound words
        words = re.findall(r'\b[\w_]+\b', sentence.lower())
        bindings = {}

        if not words:
            return bindings

        # State machine parsing
        current_role = Role.AGEN

        # Sederhana: kita asumsikan kalimat S-P-O standar
        # Kata pertama = Agen, lalu cari predikat (verb biasanya setelah agen)

        idx = 0
        while idx < len(words):
            word = words[idx]

            # Check if it's a preposition that changes the role context
            if word in self.prep_role_map:
                current_role = self.prep_role_map[word]
                idx += 1
                if idx < len(words):
                    bindings[current_role] = words[idx]
            # Simple heuristic for predicate: first word after agent that isn't a preposition
            elif Role.AGEN in bindings and Role.PREDIKAT not in bindings:
                bindings[Role.PREDIKAT] = word
                current_role = Role.PASIEN # Subsequent words default to Patient/Object
            elif current_role not in bindings:
                bindings[current_role] = word
            else:
                # Append to existing if it's an adjective/compound (simplification)
                # Like "apel" + "besar" -> "apel_besar"
                bindings[current_role] = f"{bindings[current_role]}_{word}"

            idx += 1

        # Fallback cleanup
        if Role.AGEN not in bindings and len(words) > 0:
            bindings[Role.AGEN] = words[0]

        return bindings

    def ingest_document(self, text: str) -> int:
        """
        Pipeline utama: Membaca teks, memecah kalimat, mengekstrak role,
        mendaftarkan vocab otomatis, dan meng-encode ke memori.
        """
        sentences = self._chunk_sentences(text)
        ingested_count = 0

        for sent in sentences:
            bindings = self._extract_entities_and_roles(sent)

            if not bindings or len(bindings) < 2:
                continue

            # 1. Pendaftaran Vocab Otomatis (OOTV Handling)
            for role, token in bindings.items():
                if self.engine.get_token(token) is None:
                    # Heuristik Kategori berdasarkan Role
                    if role == Role.PREDIKAT:
                        cat_guess = "aksi"
                    elif role in [Role.LOKASI, "lokasi"]:
                        cat_guess = "tempat"
                    elif role in [Role.WAKTU, "waktu"]:
                        cat_guess = "waktu"
                    elif role in [Role.INSTRUMEN, "instrumen"]:
                        cat_guess = "alat"
                    else:
                        cat_guess = "entitas"

                    self.engine.add_token(token, category=cat_guess)
                    # Automatically ensure role exists
                    if role not in self.engine.role_names:
                        self.engine.add_role(role)

            # 2. Vector Encoding & Penyimpanan (Episodic)
            encoded_vec = self.engine.encode(bindings)

            if encoded_vec is not None:
                self.engine.store_episodic(encoded_vec, metadata={"source_sentence": sent})

                # 3. Knowledge Graph Ingestion (Long-term semantic storage)
                if self.kg and Role.AGEN in bindings and Role.PREDIKAT in bindings and Role.PASIEN in bindings:
                    triple = KGTriple(
                        subject=bindings[Role.AGEN],
                        predicate=bindings[Role.PREDIKAT],
                        object=bindings[Role.PASIEN],
                        metadata={"source": "text_ingestor", "sentence": sent}
                    )
                    self.kg.ingest_triple(triple)

                ingested_count += 1

        return ingested_count

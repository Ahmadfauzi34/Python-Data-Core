import re
from typing import List, Dict, Any
from fhrr_project.core.engine import FHRREngine
from fhrr_project.core.roles import Role

class TextIngestorBlueprint:
    """
    Blueprint untuk pipeline NLP Unstructured Text to FHRR.
    Mengubah paragraf bebas menjadi FHRR phase vectors.
    """
    def __init__(self, engine: FHRREngine):
        self.engine = engine

    def _chunk_sentences(self, text: str) -> List[str]:
        """Memecah paragraf menjadi list kalimat. Bisa diganti model Spacy/NLTK nanti."""
        # Split berdasarkan titik, tanda tanya, atau seru.
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _extract_entities_and_roles(self, sentence: str) -> Dict[str, str]:
        """
        [STUB] Logika ekstraksi semantik.
        Di masa depan, gunakan Dependency Parsing (seperti spacy) untuk mendeteksi
        Subjek, Predikat, dan Objek secara akurat.
        """
        words = re.sub(r'[^\w\s]', '', sentence.lower()).split()
        bindings = {}

        # Heuristik Sangat Sederhana (Hanya untuk Blueprint):
        # Anggap Kata 1 = Agen, Kata 2 = Predikat, Kata 3+ = Pasien
        if len(words) >= 1:
            bindings[Role.AGEN] = words[0]
        if len(words) >= 2:
            bindings[Role.PREDIKAT] = words[1]
        if len(words) >= 3:
            bindings[Role.PASIEN] = words[2]

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

            if not bindings:
                continue

            # 1. Pendaftaran Vocab Otomatis (OOTV Handling)
            for role, token in bindings.items():
                # Jika token belum ada di vocabulary, tambahkan.
                # Heuristik Kategori: anggap token = kategori untuk simplifikasi.
                if self.engine.get_token(token) is None:
                    cat_guess = "kata_kerja" if role == Role.PREDIKAT else "kata_benda"
                    self.engine.add_token(token, category=cat_guess)

            # 2. Vector Encoding & Penyimpanan
            # Karena FHRREngine sudah sangat dioptimasi (Zero-Alloc Encode), ini sangat cepat!
            encoded_vec = self.engine.encode(bindings)

            if encoded_vec is not None:
                # Simpan ke memori episodik (Short-term)
                self.engine.store_episodic(encoded_vec, metadata={"source_sentence": sent})
                ingested_count += 1

        return ingested_count

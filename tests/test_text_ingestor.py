import unittest
from fhrr_project.core.engine import FHRREngine
from fhrr_project.core.roles import Role
from fhrr_project.agents.text_ingestor import TextIngestorBlueprint

class TestTextIngestorBlueprint(unittest.TestCase):
    def setUp(self):
        self.engine = FHRREngine()
        # Pre-seed the basic roles required
        self.engine.add_role(Role.AGEN)
        self.engine.add_role(Role.PREDIKAT)
        self.engine.add_role(Role.PASIEN)
        self.ingestor = TextIngestorBlueprint(self.engine)

    def test_chunking(self):
        text = "Halo dunia! Ini Budi. Budi makan apel."
        chunks = self.ingestor._chunk_sentences(text)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "Halo dunia!")
        self.assertEqual(chunks[1], "Ini Budi.")
        self.assertEqual(chunks[2], "Budi makan apel.")

    def test_extraction(self):
        sent = "Budi makan apel"
        bindings = self.ingestor._extract_entities_and_roles(sent)
        self.assertEqual(bindings[Role.AGEN], "budi")
        self.assertEqual(bindings[Role.PREDIKAT], "makan")
        self.assertEqual(bindings[Role.PASIEN], "apel")

    def test_ingest_document(self):
        doc = "Jono melempar batu. Siti menangkap bola."
        count = self.ingestor.ingest_document(doc)
        self.assertEqual(count, 2)

        # Verify tokens were auto-registered
        self.assertIsNotNone(self.engine.get_token("jono"))
        self.assertIsNotNone(self.engine.get_token("melempar"))
        self.assertIsNotNone(self.engine.get_token("bola"))

        # Verify episodic memory was populated
        self.assertTrue(len(self.engine.episodic_buffer) >= 2)

if __name__ == '__main__':
    unittest.main()

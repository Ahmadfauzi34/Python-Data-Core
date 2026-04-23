import unittest
from fhrr_project.core.engine import FHRREngine
from fhrr_project.core.roles import Role
from fhrr_project.agents.text_ingestor import TextIngestorBlueprint

class TextIngestorHeuristicsTests(unittest.TestCase):
    def setUp(self):
        self.engine = FHRREngine(dim=512)
        self.engine.add_role(Role.AGEN)
        self.engine.add_role(Role.PREDIKAT)
        self.engine.add_role(Role.PASIEN)
        self.engine.add_role(Role.LOKASI)
        self.ingestor = TextIngestorBlueprint(self.engine)

    def test_stopword_and_compound(self):
        # "yang besar" should not be the predicate. Compound limit should trigger.
        sent = "Anjing yang besar hitam manis makan tulang di taman."
        bindings = self.ingestor._extract_entities_and_roles(sent)

        # Stopword 'yang' skips to next. Predicate should be 'makan'.
        self.assertEqual(bindings.get(Role.PREDIKAT), 'makan')
        self.assertEqual(bindings.get(Role.LOKASI), 'taman')

        # Agent compound should cap out smoothly without capturing the verb
        self.assertLess(bindings.get(Role.AGEN, "").count('_'), 2)
        self.assertTrue(bindings.get(Role.AGEN, "").startswith('anjing'))

if __name__ == '__main__':
    unittest.main()

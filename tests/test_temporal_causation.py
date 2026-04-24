import unittest
import numpy as np
import tempfile
from fhrr_project.core.engine import FHRREngine
from fhrr_project.agents.text_ingestor import TextIngestorBlueprint
from fhrr_project.memory.consolidation import MetaCognitiveConsolidator
from fhrr_project.core.roles import Role

class TemporalCausationTests(unittest.TestCase):
    def setUp(self):
        self.engine = FHRREngine(dim=512)
        self.engine.add_role(Role.AGEN)
        self.engine.add_role(Role.PREDIKAT)
        self.engine.add_role(Role.LOKASI)
        self.engine.add_role(Role.PASIEN)

        self.engine.add_token("awan", "entitas")
        self.engine.add_token("mendung", "keadaan")
        self.engine.add_token("hujan", "entitas")
        self.engine.add_token("turun", "aksi")
        self.engine.add_token("tanah", "entitas")
        self.engine.add_token("basah", "keadaan")

        self.ingestor = TextIngestorBlueprint(self.engine, None)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.consolidator = MetaCognitiveConsolidator(self.engine, dataset_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_temporal_induction(self):
        # We manually inject episodic memory to ensure perfect bindings regardless of the TextIngestor heuristic limitations
        import time
        t0 = time.time()

        # Event 1: Hujan turun
        v1 = self.engine.encode({'agen': 'hujan', 'predikat': 'turun'})
        self.engine.store_episodic(v1, metadata={'timestamp': t0, 'bindings': {'agen': 'hujan', 'predikat': 'turun'}})

        # Event 2: Tanah basah
        v2 = self.engine.encode({'agen': 'tanah', 'predikat': 'basah'})
        self.engine.store_episodic(v2, metadata={'timestamp': t0 + 1, 'bindings': {'agen': 'tanah', 'predikat': 'basah'}})

        # Event 3: Hujan turun (again)
        v3 = self.engine.encode({'agen': 'hujan', 'predikat': 'turun'})
        self.engine.store_episodic(v3, metadata={'timestamp': t0 + 10, 'bindings': {'agen': 'hujan', 'predikat': 'turun'}})

        # Event 4: Jalanan basah (using basah again to form a pattern)
        v4 = self.engine.encode({'agen': 'jalanan', 'predikat': 'basah'})
        self.engine.store_episodic(v4, metadata={'timestamp': t0 + 11, 'bindings': {'agen': 'jalanan', 'predikat': 'basah'}})

        trans = self.consolidator.extract_transformations()
        rules = self.consolidator.consolidate()

        self.assertGreaterEqual(len(rules), 1)
        found = False
        for r in rules:
            if r['premise'].get('predikat') == 'turun' and r['conclusion'].get('atribut') == 'basah':
                found = True
                break
        self.assertTrue(found, "Failed to induce temporal causation rule from sequence")

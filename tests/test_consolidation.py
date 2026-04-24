import unittest
import numpy as np
import os
import tempfile
from fhrr_project.core.engine import FHRREngine
from fhrr_project.memory.consolidation import MetaCognitiveConsolidator

class ConsolidatorTests(unittest.TestCase):
    def setUp(self):
        self.engine = FHRREngine(dim=512)
        self.engine.add_token("baca", "aksi")
        self.engine.add_token("pintar", "keadaan")
        self.engine.add_token("belajar", "aksi")
        self.engine.add_role("predikat")
        self.engine.add_role("atribut")

        self.temp_dir = tempfile.TemporaryDirectory()
        self.consolidator = MetaCognitiveConsolidator(self.engine, dataset_dir=self.temp_dir.name)
        self.staging_file = os.path.join(self.temp_dir.name, "reasoning_patterns.auto.yaml")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_consolidation_rule_induction(self):
        # Seed 3 highly similar transforms
        self.engine.learn_transform_from_data("t1", "baca", "pintar")

        # Force "belajar" to be identical to "baca" phase-wise for guaranteed clustering
        idx_baca = self.engine._token_name_to_idx["baca"]
        idx_belajar = self.engine._token_name_to_idx["belajar"]
        self.engine.token_phases[idx_belajar] = self.engine.token_phases[idx_baca]

        self.engine.learn_transform_from_data("t2", "belajar", "pintar")

        # Test Dry Run capability (should not write to disk)
        from fhrr_project.core.runner import FHRRResearchRunner
        runner = FHRRResearchRunner(dim=512)
        runner.consolidator = self.consolidator
        dry_rules = runner.sleep_and_consolidate(dry_run=True)
        self.assertGreaterEqual(len(dry_rules), 1)
        self.assertFalse(os.path.exists(self.staging_file))

        rules = self.consolidator.consolidate()
        self.assertGreaterEqual(len(rules), 1)
        self.assertIn("auto_induced_", rules[0]['name'])

        # Test dedup persistence
        self.consolidator.persist_rules_to_dataset(rules)
        self.assertTrue(os.path.exists(self.staging_file))

        # Test idempotency (calling again shouldn't write duplicate rules)
        rules2 = self.consolidator.consolidate()
        self.consolidator.persist_rules_to_dataset(rules2)

        with open(self.staging_file, 'r') as f:
            content = f.read()
            # The rule name should only appear once if dedup by signature is working
            self.assertEqual(content.count(rules[0]['semantic_signature']), 1)

if __name__ == '__main__':
    unittest.main()

"""
Smoke test: dataset loading, KG ingest, dan QA inference end-to-end.

Run:
    python -m unittest tests.test_smoke -v
"""
from __future__ import annotations
import os
import sys
import unittest

# Pastikan fhrr_project on sys.path supaya import-relatif (core., data., dst.) jalan.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "fhrr_project"))
sys.path.insert(0, _ROOT)

from fhrr_core import (  # noqa: E402
    FHRRResearchRunner, extend_engine_open_vocab,
    KnowledgeGraphIngestor, FHRRTopologicalLayer,
    SelfSupervisedDiscovery, SelfImprovementEngine,
    FHRRQueryInterface, load_dataset, list_datasets,
    ingest_dataset_to_kg, validate_dataset,
)


def _build_api(dim: int = 1024):
    """Bangun pipeline lengkap (engine + KG + topo + api). Dim kecil supaya cepat."""
    ds = load_dataset("default", strict=False)
    runner = FHRRResearchRunner(dim=dim)
    runner.load_dataset(ds)
    open_vocab = extend_engine_open_vocab(runner.engine)
    kg = KnowledgeGraphIngestor(runner.engine, open_vocab)
    n_triples = ingest_dataset_to_kg(kg, ds)
    discoverer = SelfSupervisedDiscovery(runner.engine, window_size=3)
    topo = FHRRTopologicalLayer(runner.engine)
    runner.attach_topology(topo)
    improver = SelfImprovementEngine(runner.engine, topo, discoverer, kg)
    api = FHRRQueryInterface(runner)
    api.attach_kg(kg).attach_discoverer(discoverer).attach_improver(improver)
    return api, ds, n_triples


class DatasetTests(unittest.TestCase):
    def test_default_dataset_loads(self):
        ds = load_dataset("default", strict=False)
        self.assertIn("vocab", ds)
        self.assertGreater(len(ds["observations"]), 0)
        self.assertGreater(len(ds["qa_pairs"]), 0)

    def test_default_dataset_has_no_errors(self):
        """Default dataset boleh punya warning, tapi TIDAK boleh error."""
        ds = load_dataset("default", strict=False)
        issues = validate_dataset(ds)
        errors = [i for i in issues if i.severity == "error"]
        self.assertEqual(errors, [], f"Dataset default punya error: {errors}")

    def test_validator_catches_dangling_qa_source(self):
        bad = {
            "vocab": {"categories": {"x": ["a"]}},
            "observations": [{"id": "obs1", "bindings": {"agen": "a"}}],
            "qa_pairs": [{"id": "qa1", "source": "obs_NOT_EXIST", "answer_role": "agen"}],
        }
        issues = validate_dataset(bad)
        errs = [i for i in issues if i.severity == "error"]
        self.assertTrue(any("source" in e.where for e in errs), f"Tidak deteksi dangling source: {errs}")

    def test_validator_catches_duplicate_obs_id(self):
        bad = {
            "vocab": {"categories": {}},
            "observations": [
                {"id": "dup", "bindings": {"agen": "a"}},
                {"id": "dup", "bindings": {"agen": "b"}},
            ],
        }
        errs = [i for i in validate_dataset(bad) if i.severity == "error"]
        self.assertTrue(any("duplikat" in e.message for e in errs))

    def test_list_datasets_includes_default(self):
        self.assertIn("default", list_datasets())


class IngestTests(unittest.TestCase):
    def test_ingest_produces_triples(self):
        _, _, n = _build_api()
        # 10 observation, masing-masing minimal 1 triple inti → setidaknya 10.
        self.assertGreaterEqual(n, 10)


class QueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api, cls.ds, _ = _build_api()

    def test_who_query_returns_known_agent(self):
        # "siapa makan mangga" — minimal jawaban harus ada di list agen di dataset.
        result = self.api.ask("siapa makan mangga", explain=False)
        self.assertTrue(result.answer)
        self.assertNotEqual(result.mechanism, "unparseable")

    def test_unparseable_query_does_not_crash(self):
        result = self.api.ask("xyz qwerty foobar", explain=False)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.mechanism)


if __name__ == "__main__":
    unittest.main()

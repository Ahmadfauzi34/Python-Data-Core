import unittest
import os
import tempfile
import shutil
import yaml
from fhrr_project.data.loader import load_dataset, _deep_merge

class TestLoader(unittest.TestCase):
    def test_deep_merge(self):
        target = {
            "vocab": {"categories": {"hewan": ["kucing"]}},
            "observations": [{"id": 1}]
        }
        source = {
            "vocab": {"categories": {"hewan": ["anjing"], "benda": ["batu"]}},
            "observations": [{"id": 2}]
        }
        res = _deep_merge(target, source)

        # Verify lists were extended
        self.assertEqual(res["vocab"]["categories"]["hewan"], ["kucing", "anjing"])
        self.assertEqual(res["vocab"]["categories"]["benda"], ["batu"])
        self.assertEqual(len(res["observations"]), 2)

    def test_load_multi_file_directory(self):
        # Create a temporary directory structure simulating _DATASETS_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            ds_dir = os.path.join(temp_dir, "my_dataset")
            os.makedirs(ds_dir)

            # File 1: Vocab
            f1 = {"vocab": {"categories": {"test_cat": ["token1"]}}}
            with open(os.path.join(ds_dir, "1_vocab.yaml"), "w") as f:
                yaml.dump(f1, f)

            # File 2: Observations
            f2 = {"observations": [{"bindings": {"agen": "token1"}}]}
            with open(os.path.join(ds_dir, "2_obs.yaml"), "w") as f:
                yaml.dump(f2, f)

            # Now patch the _DATASETS_DIR path temporarily or use absolute path
            # Since load_dataset accepts absolute path if it escapes _DATASETS_DIR exception?
            # Actually _resolve_path restricts to is_relative_to(base_dir).
            # We must test using the existing `fhrr_project/data/datasets` folder.

            # Alternative: Since we can't easily mock the global var here without monkeypatching,
            # let's just make a dummy dir inside the actual datasets folder.
            actual_ds_dir = os.path.join("fhrr_project", "data", "datasets", "dummy_test_ds")
            try:
                os.makedirs(actual_ds_dir, exist_ok=True)
                with open(os.path.join(actual_ds_dir, "1_vocab.yaml"), "w") as f:
                    yaml.dump(f1, f)
                with open(os.path.join(actual_ds_dir, "2_obs.yaml"), "w") as f:
                    yaml.dump(f2, f)

                data = load_dataset("dummy_test_ds", strict=False) # strict=False to bypass schema rules for partial dummy data

                self.assertIn("test_cat", data["vocab"]["categories"])
                self.assertEqual(data["vocab"]["categories"]["test_cat"], ["token1"])
                self.assertEqual(len(data["observations"]), 1)
                self.assertEqual(data["observations"][0]["bindings"]["agen"], "token1")
            finally:
                if os.path.exists(actual_ds_dir):
                    shutil.rmtree(actual_ds_dir)

if __name__ == '__main__':
    unittest.main()

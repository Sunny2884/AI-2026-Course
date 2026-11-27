import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("tiled_pipeline", ROOT / "tools/run_tiled_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def record(image_id=17):
    mask = np.zeros((1024, 1024), dtype=np.uint8, order="F"); mask[5:7, 9:11] = 1
    rle = mu.encode(mask); rle["counts"] = rle["counts"].decode()
    return {"image_id": image_id, "category_id": 1, "score": .8,
            "bbox": [9., 5., 2., 2.], "segmentation": rle}


class TiledPipelineTests(unittest.TestCase):
    def test_gpu_selection_excludes_busy_and_low_memory(self):
        snapshot = [{"index": 0, "free_mib": 37000, "utilization": 99},
                    {"index": 1, "free_mib": 39000, "utilization": 30},
                    {"index": 2, "free_mib": 31000, "utilization": 10},
                    {"index": 3, "free_mib": 8000, "utilization": 0}]
        with patch.object(pipeline, "gpu_snapshot", return_value=snapshot), patch.object(pipeline.time, "sleep"):
            chosen, _ = pipeline.choose_gpus("auto", 3)
            self.assertEqual(chosen, [2, 1])
            with self.assertRaises(RuntimeError):
                pipeline.choose_gpus("3", 3)

    def test_atomic_cache_provenance_and_image_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "17.json.gz"
            item = {"image_id": 17, "signature": "a", "global": [record()], "tiles5": [record()]}
            pipeline.write_item(path, item)
            self.assertEqual(pipeline.read_item(path, 17, "a"), item)
            self.assertFalse(path.with_suffix(".tmp").exists())
            with self.assertRaises(ValueError):
                pipeline.read_item(path, 18, "a")
            with self.assertRaises(ValueError):
                pipeline.read_item(path, 17, "different_weights")

    def test_submission_order_and_csv_json_consistency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); template = root / "template.csv"
            template.write_text('image_id,PredictionString\n18,[]\n17,[]\n')
            rec = record(); obj = {key: rec[key] for key in ["category_id", "score", "segmentation"]}
            (root / "submission.json").write_text(json.dumps([rec]))
            with (root / "submission.csv").open("w", newline="") as stream:
                writer = csv.writer(stream); writer.writerow(["image_id", "PredictionString"])
                writer.writerow([18, "[]"]); writer.writerow([17, json.dumps([obj])])
            result = pipeline.verify_submission(root, template, [17, 18])
            self.assertEqual(result["images"], 2); self.assertEqual(result["instances"], 1)
            rec["score"] = .2
            (root / "submission.json").write_text(json.dumps([rec]))
            with self.assertRaises(AssertionError):
                pipeline.verify_submission(root, template, [17, 18])

    def test_equivalence_detects_geometry_changes(self):
        rec = record(); copy = json.loads(json.dumps(rec)); copy["score"] += 1e-6
        pipeline.check_equivalent([copy], [rec])
        copy["bbox"][0] += 1
        with self.assertRaises(AssertionError):
            pipeline.check_equivalent([copy], [rec])


if __name__ == "__main__":
    unittest.main()

"""Numerical and geometry checks for instance scoring, without a model forward."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from pycocotools import mask as mu
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2f_rgbd.postprocess import decode_records, decode_record_variants


class PostprocessTests(unittest.TestCase):
    def example(self):
        # Equal class scores; the second query has the more confident mask.
        masks = torch.full((1, 3, 2, 3), -3.0, dtype=torch.bfloat16)
        masks[0, 0, :, 0] = 0.25
        masks[0, 1, :, 1] = 3.0
        return SimpleNamespace(class_queries_logits=torch.tensor([[[8., 0.]] * 3], dtype=torch.bfloat16),
                               masks_queries_logits=masks)

    def test_quality_matches_formula_and_preserves_geometry(self):
        out = self.example()
        pure = decode_records(out, [7], output_size=(2, 3), score_mode="class_fp32")
        scored = decode_records(out, [7], output_size=(2, 3), score_mode="mask_quality")
        self.assertEqual(len(scored), 2)  # The entirely empty query is removed.
        for original, new in zip(pure, scored):
            self.assertEqual(original["segmentation"], new["segmentation"])
            self.assertEqual(original["bbox"], new["bbox"])
        self.assertGreater(scored[1]["score"], scored[0]["score"])
        self.assertAlmostEqual(scored[0]["score"], pure[0]["score"] * torch.tensor(.25).sigmoid().item(), places=6)
        self.assertEqual(scored[0]["bbox"], [0., 0., 1., 2.])
        self.assertEqual(mu.area(scored[0]["segmentation"]), 2)

    def test_legacy_reproduces_old_bfloat16_decoder(self):
        out = self.example()
        expected_scores = out.class_queries_logits.softmax(-1)[0, :, 0]
        expected_masks = F.interpolate(out.masks_queries_logits, size=(5, 7), mode="bilinear",
                                       align_corners=False).sigmoid() > .5
        decoded = decode_records(out, [3], output_size=(5, 7), score_mode="legacy_bf16")
        for index, record in enumerate(decoded):
            self.assertEqual(record["score"], expected_scores[index].item())
            np.testing.assert_array_equal(mu.decode(record["segmentation"]), expected_masks[0, index].numpy())

    def test_chunking_does_not_change_results(self):
        out = self.example()
        for mode in ["legacy_bf16", "class_fp32", "mask_quality"]:
            self.assertEqual(decode_records(out, [9], output_size=8, score_mode=mode, chunk_size=1),
                             decode_records(out, [9], output_size=8, score_mode=mode, chunk_size=16))

    def test_shared_geometry_matches_independent_decoders(self):
        torch.manual_seed(71)
        for shape in [(2, 3), (8, 5)]:
            out = SimpleNamespace(class_queries_logits=torch.randn(2, 7, 2).bfloat16(),
                                  masks_queries_logits=torch.randn(2, 7, *shape).bfloat16())
            variants = decode_record_variants(out, [5, 6], output_size=(11, 17))
            for mode, records in variants.items():
                self.assertEqual(records, decode_records(out, [5, 6], output_size=(11, 17), score_mode=mode))

    def test_empty_queries_and_invalid_values(self):
        out = self.example()
        self.assertEqual(decode_records(out, [1], score_threshold=1.0, output_size=8), [])
        out.class_queries_logits[0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            decode_records(out, [1], output_size=8)


if __name__ == "__main__":
    unittest.main()

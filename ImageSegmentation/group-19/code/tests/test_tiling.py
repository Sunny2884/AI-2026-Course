from pathlib import Path
import sys
import unittest

import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2f_rgbd.tiling import map_tile_record, mask_nms, max_matching_count, tile_positions


def record(mask, score=.9):
    rle = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode()
    return {"image_id": 1, "category_id": 1, "score": score,
            "bbox": mu.toBbox(rle).tolist(), "segmentation": rle}


class TilingTests(unittest.TestCase):
    def test_positions_cover_image(self):
        self.assertEqual(tile_positions(1024, 384, 320), [0, 320, 640])
        self.assertEqual(tile_positions(1024, 512, 300), [0, 300, 512])
        self.assertEqual(tile_positions(1024, 1024, 1024), [0])

    def test_mapping_and_edge_rejection(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[3:5, 3:5] = 1
        mapped = map_tile_record(record(mask), 4, 4, 8, full_size=16)
        self.assertEqual(mapped["bbox"], [7., 7., 2., 2.])
        self.assertEqual(int(mu.area(mapped["segmentation"])), 4)
        np.testing.assert_array_equal(mu.decode(mapped["segmentation"])[4:12, 4:12], mask)
        mask[:] = 0; mask[0:2, 3:5] = 1
        self.assertIsNone(map_tile_record(record(mask), 4, 4, 8, full_size=16))
        self.assertIsNotNone(map_tile_record(record(mask), 4, 0, 8, full_size=16))

    def test_mask_nms_keeps_disjoint_interleaved_masks(self):
        a = np.zeros((8, 8), dtype=np.uint8); a[1:7, 1] = 1; a[1, 1:7] = 1
        b = np.zeros((8, 8), dtype=np.uint8); b[2:7, 6] = 1; b[6, 2:7] = 1
        kept = mask_nms([record(a, .9), record(a, .8), record(b, .7)])
        self.assertEqual([r["score"] for r in kept], [.9, .7])

    def test_oracle_matches_each_prediction_once(self):
        self.assertEqual(max_matching_count(np.array([[.8, .8]]), .5), 1)
        self.assertEqual(max_matching_count(np.array([[.8, .1], [.1, .8]]), .75), 2)


if __name__ == "__main__":
    unittest.main()

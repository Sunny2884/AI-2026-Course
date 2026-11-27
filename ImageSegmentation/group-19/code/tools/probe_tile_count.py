"""CPU-only layout ablation on frozen 1.5 global and 9-tile predictions."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2f_rgbd.data import RGBDInstanceDataset
from m2f_rgbd.engine import evaluate_predictions
from m2f_rgbd.tiling import candidate_recall, map_tile_record, mask_nms
from m2f_rgbd.utils import project_path, write_json

# Fixed geometric layouts, never selected using GT. Indices are row-major.
LAYOUTS = {
    "tiles4_corners": [0, 2, 6, 8],
    "tiles5_corners_center": [0, 2, 4, 6, 8],
    "tiles6_left_right": [0, 2, 3, 5, 6, 8],
    "tiles6_top_bottom": [0, 1, 2, 6, 7, 8],
    "tiles9_all": list(range(9)),
}


def read_gzip(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def by_image(records):
    result = defaultdict(list)
    for record in records:
        result[record["image_id"]].append(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_initialized(), "This experiment must not initialize CUDA"
    torch.set_num_threads(2)
    base, scale, out = map(lambda p: Path(p).resolve(), [args.base, args.scale, args.output])
    frozen = json.loads((base / "manifest.json").read_text())
    scaled = json.loads((scale / "manifest.json").read_text())
    assert frozen["images"] == scaled["image_ids"] and len(frozen["images"]) == 32
    assert frozen["checkpoint_sha256"] == scaled["checkpoint_sha256"]
    for name in ["m2f_rgbd/data.py", "m2f_rgbd/tiling.py"]:
        assert hashlib.sha256(project_path(name).read_bytes()).hexdigest() == frozen["source_sha256"][name]
    out.mkdir(exist_ok=False)
    positions = [(x, y) for y in frozen["tile_positions"] for x in frozen["tile_positions"]]
    coverage = {}
    for name, indices in LAYOUTS.items():
        pixels = np.zeros((1024, 1024), dtype=bool)
        for index in indices:
            x, y = positions[index]
            pixels[y:y + frozen["tile_original_size"], x:x + frozen["tile_original_size"]] = True
        coverage[name] = float(pixels.mean())
    write_json(out / "manifest.json", {
        "base_manifest": str(base / "manifest.json"), "scale_manifest": str(scale / "manifest.json"),
        "image_ids": frozen["images"], "layouts": LAYOUTS, "tile_positions": positions,
        "tile_size": frozen["tile_original_size"], "global_scale": 1.5,
        "mask_nms_iou": .5, "maxDets": [1, 10, 100], "new_forward_count": 0,
        "retained_increment_gate": .90, "selection_basis": "Fixed layouts on existing validation scenes",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    ds = RGBDInstanceDataset(project_path("dataset"), "val", jitter=None, crop_size=None)
    truth = ds.build_coco(); annotations = defaultdict(list)
    for ann in truth.dataset["annotations"]:
        if not ann.get("iscrowd", 0):
            annotations[ann["image_id"]].append(ann)
    previous9 = by_image(read_gzip(scale / "s150_plus_tiles.json.gz"))
    global_records = read_gzip(scale / "s150_global_box_nms.json.gz")
    predictions = {name: [] for name in LAYOUTS}
    recalls = {name: {group: defaultdict(int) for group in ["small", "tiny"]} for name in LAYOUTS}
    started = time.monotonic()
    for number, image_id in enumerate(frozen["images"], 1):
        global150 = read_gzip(scale / "views" / f"{image_id}_global150.json.gz")
        tiles = []
        for x, y in positions:
            local = read_gzip(base / "views" / f"{image_id}_tile_{x}_{y}.json.gz")
            mapped = [map_tile_record(record, x, y, frozen["tile_original_size"]) for record in local]
            tiles.append([record for record in mapped if record is not None])
        for name, indices in LAYOUTS.items():
            union = list(global150)
            for index in indices:
                union.extend(tiles[index])
            records = mask_nms(union)
            if name == "tiles9_all":
                assert records == previous9[image_id], "Frozen 9-tile reference changed"
            predictions[name].extend(records)
            diagnostic = candidate_recall(records, annotations[image_id])
            for group, values in diagnostic.items():
                for key, value in values.items():
                    recalls[name][group][key] += value
        if number % 8 == 0:
            print(f"Merged {number}/32 cached images; zero new forwards", flush=True)
    metrics = {}
    metrics["global150_only"] = evaluate_predictions(truth, global_records, frozen["images"])
    for name in LAYOUTS:
        metrics[name] = evaluate_predictions(truth, predictions[name], frozen["images"])
        with gzip.open(out / f"{name}.json.gz", "wt") as stream:
            json.dump(predictions[name], stream, allow_nan=False)
        print(name, "AP_s=", metrics[name]["segm"]["AP_s"], flush=True)
    baseline = metrics["global150_only"]["segm"]["AP_s"]
    full = metrics["tiles9_all"]["segm"]["AP_s"]
    results = {}
    for name, indices in LAYOUTS.items():
        aps = metrics[name]["segm"]["AP_s"]
        normalized = {group: {**values, **{key.replace("matched_", "recall_"): value / max(values["gt"], 1)
                                          for key, value in values.items() if key.startswith("matched_")}}
                      for group, values in recalls[name].items()}
        retained = (aps - baseline) / max(full - baseline, 1e-9)
        results[name] = {"tiles": len(indices), "views_per_image": 1 + len(indices),
                         "AP_s": aps, "Mask_AP": metrics[name]["segm"]["AP"],
                         "AP_s_difference_vs9": aps - full, "retained_AP_s_fraction": aps / full,
                         "retained_tile_increment_fraction": retained,
                         "passes_retained_increment_gate": retained >= .90,
                         "local_highres_area_coverage": coverage[name],
                         "nominal_forward_count_saving_vs9": 1 - (1 + len(indices)) / 10,
                         "nominal_input_pixel_work_saving_vs9": 1 - (2.25 + len(indices)) / 11.25,
                         "recall": normalized}
    eligible = [name for name in results if name != "tiles9_all" and results[name]["passes_retained_increment_gate"]]
    recommended = min(eligible, key=lambda name: (results[name]["tiles"], -results[name]["AP_s"])) if eligible else "tiles9_all"
    assert not torch.cuda.is_initialized()
    summary = {"images": 32, "new_forwards": 0, "GPU_initialized": False,
               "global150_AP_s": baseline, "tiles9_AP_s": full, "results": results,
               "recommendation_by_predeclared_gate": recommended,
               "gate": "Retain >=90% of the AP_s increment provided by 9 tiles over global150",
               "elapsed_seconds": time.monotonic() - started,
               "caveats": ["Only 32 reused validation scenes; layout selection needs independent validation",
                           "448x448 corner/edge subsets leave gaps in high-resolution coverage; global branch covers the whole image",
                           "Savings are counts/pixel-work estimates, not measured deployment latency"]}
    write_json(out / "summary.json", summary)
    write_json(out / "metrics.json", metrics)
    print(json.dumps(summary, indent=2), flush=True)
    print("CPU_TILE_COUNT_ABLATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

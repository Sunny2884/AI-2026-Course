"""Offline paired evaluation: identity, pose, static AUs and MUSIQ (no LPIPS)."""

import argparse
import csv
import hashlib
import io
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "csv", "generated", "output", "insightface-root", "musiq-weight", "hsemotion-weight"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--openface-bin", type=Path)
    parser.add_argument("--skip-openface", action="store_true", help="Only for an explicitly incomplete benchmark.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--det-threshold", type=float, default=0.5)
    parser.add_argument("--det-size", type=int, default=320)
    parser.add_argument("--au-confidence", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def statistics(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p90": None}
    if not np.isfinite(values).all():
        raise ValueError("Non-finite metric encountered")
    return {"count": len(values), "mean": float(values.mean()),
            "median": float(np.median(values)), "p90": float(np.quantile(values, 0.9))}


def load_image(path, size, quality=None):
    with Image.open(path) as source:
        image = source.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    if quality is not None:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            image = decoded.convert("RGB")
    return np.asarray(image).copy()


class FaceMetrics:
    def __init__(self, root, device, threshold, det_size):
        import onnxruntime as ort
        from insightface.model_zoo import get_model

        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("ONNX Runtime CUDA provider is unavailable")
            providers.insert(0, "CUDAExecutionProvider")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.detector, self.identity, self.landmark = [
            get_model(str(root / filename), providers=providers, sess_options=options)
            for filename in ("det_10g.onnx", "w600k_r50.onnx", "1k3d68.onnx")
        ]
        if any(model is None for model in (self.detector, self.identity, self.landmark)):
            raise RuntimeError("Could not route an InsightFace ONNX model")
        ctx = 0 if device == "cuda" else -1
        self.detector.prepare(ctx_id=ctx, input_size=(det_size, det_size), det_thresh=threshold)
        self.identity.prepare(ctx_id=ctx)
        self.landmark.prepare(ctx_id=ctx)
        self.providers = self.detector.session.get_providers()
        if device == "cuda" and self.providers[0] != "CUDAExecutionProvider":
            raise RuntimeError("Requested CUDA, but ONNX Runtime fell back to CPU")

    def extract(self, rgb):
        from insightface.app.common import Face

        bgr = rgb[:, :, ::-1].copy()
        boxes, keypoints = self.detector.detect(bgr, max_num=0)
        if len(boxes) == 0:
            return {"face_count": len(boxes)}
        # Use the largest detected face as the primary subject. Background
        # faces and duplicate detections are retained in face_count but do
        # not invalidate the primary-face metric.
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        primary_index = int(np.argmax(areas))
        face = Face(bbox=boxes[primary_index, :4], kps=keypoints[primary_index],
                    det_score=float(boxes[primary_index, 4]))
        self.identity.get(bgr, face)
        self.landmark.get(bgr, face)
        embedding = np.asarray(face.embedding, dtype=float).ravel()
        norm = float(np.linalg.norm(embedding))
        pose = np.asarray(face.pose, dtype=float)
        if norm <= 0 or not np.isfinite(embedding).all() or not np.isfinite(pose).all():
            raise ValueError("Invalid identity embedding or pose")
        return {"face_count": len(boxes), "embedding": embedding / norm,
                "pose": pose, "det_score": face.det_score,
                "bbox": np.asarray(face.bbox, dtype=float)}


def crop_primary_face(rgb, bbox, padding=0.35):
    """Crop one primary face with context for OpenFace AU estimation."""
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(width, int(x2 + pad_x))
    bottom = min(height, int(y2 + pad_y))
    crop = rgb[top:bottom, left:right]
    if crop.size == 0:
        return rgb
    return np.asarray(Image.fromarray(crop).resize((256, 256), Image.Resampling.LANCZOS))


class HSEmotionMetrics:
    """Load the trusted local HSEmotion checkpoint and compare face crops."""

    def __init__(self, weight, device):
        import hsemotion.facial_emotions as facial_emotions

        original_load = facial_emotions.torch.load
        def trusted_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)
        facial_emotions.torch.load = trusted_load
        facial_emotions.get_model_path = lambda _name: str(weight)
        self.model = facial_emotions.HSEmotionRecognizer(
            model_name="enet_b0_8_va_mtl", device=device
        )

    def extract(self, image):
        features = self.model.extract_features(image)[0].astype(np.float32)
        norm = float(np.linalg.norm(features))
        if norm <= 0 or not np.isfinite(features).all():
            raise ValueError("Invalid HSEmotion embedding")
        scores = self.model.get_probab(features[None])[0]
        if not np.isfinite(scores).all():
            raise ValueError("Invalid HSEmotion output")
        expression = scores[:-2]
        probabilities = np.exp(expression - np.max(expression))
        probabilities /= probabilities.sum()
        return {"embedding": features / norm, "va": scores[-2:].astype(float),
                "probabilities": probabilities}


def openface_features(binary, input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    log = output_dir / "openface.log"
    command = [str(binary.resolve()), "-fdir", str(input_dir.resolve()),
               "-out_dir", str(output_dir.resolve()), "-aus", "-pose"]
    with log.open("w") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True,
                       cwd=binary.resolve().parent)
    return output_dir


def read_aus(directory, image_id, threshold):
    path = directory / (image_id + ".csv")
    if not path.is_file():
        return None, "missing_csv"
    rows = read_csv(path)
    if len(rows) != 1:
        return None, "face_count_{}".format(len(rows))
    row = rows[0]
    # FaceLandmarkImg omits the sequence-only success column.
    if ("success" in row and float(row["success"]) != 1) or float(row.get("confidence", "0")) < threshold:
        return None, "tracking_failed_or_low_confidence"
    values = {name: float(value) for name, value in row.items()
              if name.startswith("AU") and name.endswith("_r")}
    if not values or not all(math.isfinite(v) for v in values.values()):
        return None, "invalid_au_vector"
    return values, "ok"


def main():
    args = arguments()
    started = time.perf_counter()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use CPU on codelab or run a GPU job")
    if args.size < 64 or args.threads < 1 or not 1 <= args.jpeg_quality <= 100:
        raise ValueError("Invalid resolution, thread count or JPEG quality")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index or shard count")
    if not args.skip_openface and (not args.openface_bin or not args.openface_bin.is_file()):
        raise FileNotFoundError("OpenFace FaceLandmarkImg binary is required")
    for path in [args.musiq_weight, args.hsemotion_weight] + [args.insightface_root / name for name in
                 ("det_10g.onnx", "w600k_r50.onnx", "1k3d68.onnx")]:
        if not path.is_file():
            raise FileNotFoundError(path)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    rows = read_csv(args.csv)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or any(Path(i).name != i for i in ids):
        raise ValueError("CSV must contain unique, safe image IDs")
    generated = {}
    for path in args.generated.iterdir():
        if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            if path.stem in generated:
                raise ValueError("Multiple generated images for " + path.stem)
            generated[path.stem] = path
    if set(ids) != set(generated):
        raise ValueError("Generated IDs mismatch: missing={}, extra={}".format(
            sorted(set(ids) - set(generated))[:5], sorted(set(generated) - set(ids))[:5]))
    rows = rows[args.shard_index::args.num_shards]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("No images selected for evaluation")
    args.output.mkdir(parents=True, exist_ok=True)
    input_dirs = {side: args.output / "inputs" / side for side in ("original", "generated")}
    for directory in input_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
        unexpected = {p.stem for p in directory.glob("*.png")} - {r["id"] for r in rows}
        if unexpected:
            raise ValueError("Output directory contains images from a different run; choose a fresh output")

    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    config["csv_sha256"] = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    config["musiq_sha256"] = hashlib.sha256(args.musiq_weight.read_bytes()).hexdigest()
    config["hsemotion_sha256"] = hashlib.sha256(args.hsemotion_weight.read_bytes()).hexdigest()
    config["expression_mode"] = "OpenFace 2.2 static image AU regression"
    config["pose_order"] = ["pitch", "yaw", "roll"]
    config["package_versions"] = {name: importlib.metadata.version(name) for name in
                                  ("torch", "pyiqa", "insightface", "onnxruntime", "hsemotion")}
    json_write(args.output / "config.json", config)

    import pyiqa
    face_metrics = FaceMetrics(args.insightface_root, args.device, args.det_threshold, args.det_size)
    quality = pyiqa.create_metric("musiq", device=args.device,
                                 pretrained_model_path=str(args.musiq_weight)).eval()
    hsemotion = HSEmotionMetrics(args.hsemotion_weight, args.device)
    model_loaded = time.perf_counter()
    records = []
    with (args.output / "progress.jsonl").open("w") as progress:
        for index, row in enumerate(rows):
            image_id = row["id"]
            original = load_image(args.data_root / row["image_path"], args.size)
            anonymized = load_image(generated[image_id], args.size, args.jpeg_quality)
            images = {"original": original, "generated": anonymized}
            record = {"id": image_id, "original_path": row["image_path"],
                      "generated_path": str(generated[image_id]), "errors": []}
            features = {}
            primary_images = {}
            for side, image in images.items():
                try:
                    features[side] = face_metrics.extract(image)
                    record[side + "_face_count"] = features[side]["face_count"]
                    if features[side]["face_count"]:
                        image_for_openface = crop_primary_face(image, features[side]["bbox"])
                    else:
                        image_for_openface = image
                except Exception as error:
                    features[side] = {"face_count": -1}
                    record[side + "_face_count"] = -1
                    image_for_openface = image
                    record["errors"].append("{}: {}: {}".format(side, type(error).__name__, error))
                Image.fromarray(image_for_openface).save(input_dirs[side] / (image_id + ".png"))
                primary_images[side] = image_for_openface
                tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(args.device) / 255
                with torch.inference_mode():
                    score = float(quality(tensor).item())
                if not math.isfinite(score):
                    raise ValueError("Non-finite MUSIQ score: " + image_id)
                record["musiq_" + side] = score
            record["identity_cosine"] = None
            record["pose_mae_deg"] = None
            if all(features[side]["face_count"] > 0 for side in images):
                original_face, generated_face = features["original"], features["generated"]
                record["identity_cosine"] = float(np.clip(
                    np.dot(original_face["embedding"], generated_face["embedding"]), -1, 1))
                delta = (generated_face["pose"] - original_face["pose"] + 180) % 360 - 180
                record["pose_mae_deg"] = float(np.abs(delta).mean())
                for axis, angle in zip(("pitch", "yaw", "roll"), np.abs(delta)):
                    record[axis + "_error_deg"] = float(angle)
            try:
                hse = {side: hsemotion.extract(primary_images[side]) for side in images}
                record["hsemotion_cosine"] = float(np.clip(
                    np.dot(hse["original"]["embedding"], hse["generated"]["embedding"]), -1, 1))
                record["hsemotion_va_mae"] = float(np.abs(
                    hse["generated"]["va"] - hse["original"]["va"]).mean())
                record["hsemotion_original_class"] = int(np.argmax(hse["original"]["probabilities"]))
                record["hsemotion_generated_class"] = int(np.argmax(hse["generated"]["probabilities"]))
            except Exception as error:
                record["hsemotion_cosine"] = None
                record["hsemotion_va_mae"] = None
                record["hsemotion_error"] = "{}: {}".format(type(error).__name__, error)
            record["musiq_delta"] = record["musiq_generated"] - record["musiq_original"]
            records.append(record)
            progress.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            progress.flush()
            if (index + 1) % 10 == 0 or index + 1 == len(rows):
                elapsed = time.perf_counter() - model_loaded
                print("paired metrics {}/{}; {:.2f} s/pair".format(index + 1, len(rows), elapsed / (index + 1)), flush=True)
    paired_finished = time.perf_counter()

    au_columns = None
    if not args.skip_openface:
        au_dirs = {side: openface_features(args.openface_bin, directory, args.output / "openface" / side)
                   for side, directory in input_dirs.items()}
        for record in records:
            values = {}
            record["au_mae"] = None
            for side, directory in au_dirs.items():
                values[side], status = read_aus(directory, record["id"], args.au_confidence)
                record["au_" + side + "_status"] = status
            if all(values.values()):
                if set(values["original"]) != set(values["generated"]):
                    raise ValueError("AU columns differ between original and generated")
                columns = sorted(values["original"])
                if au_columns is not None and columns != au_columns:
                    raise ValueError("AU columns differ between images")
                au_columns = columns
                for column in columns:
                    record[column + "_error"] = abs(values["original"][column] - values["generated"][column])
                record["au_mae"] = float(np.mean([record[c + "_error"] for c in columns]))
    finished = time.perf_counter()
    metric_names = ["identity_cosine", "pose_mae_deg", "pitch_error_deg", "yaw_error_deg", "roll_error_deg",
                    "musiq_original", "musiq_generated", "musiq_delta", "au_mae",
                    "hsemotion_cosine", "hsemotion_va_mae"]
    summary = {
        "complete_protocol": not args.skip_openface,
        "complete_split": len(records) == len(ids),
        "csv_total": len(ids),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "pairs": len(records), "evaluated_size": args.size,
        "single_face_rate_original": sum(r["original_face_count"] == 1 for r in records) / len(records),
        "single_face_rate_generated": sum(r["generated_face_count"] == 1 for r in records) / len(records),
        "multi_face_rate_original": sum(r["original_face_count"] > 1 for r in records) / len(records),
        "multi_face_rate_generated": sum(r["generated_face_count"] > 1 for r in records) / len(records),
        "valid_identity_pairs": sum(r["identity_cosine"] is not None for r in records),
        "valid_au_pairs": sum(r.get("au_mae") is not None for r in records),
        "valid_hsemotion_pairs": sum(r.get("hsemotion_cosine") is not None for r in records),
        "onnx_providers": face_metrics.providers,
        "au_columns": au_columns,
        "metrics": {name: statistics([r.get(name) for r in records]) for name in metric_names},
        "timing_seconds": {"load": model_loaded - started, "paired": paired_finished - model_loaded,
                           "openface": finished - paired_finished, "total": finished - started},
    }
    json_write(args.output / "summary.json", summary)
    columns = sorted(set().union(*(record.keys() for record in records)))
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "errors": json.dumps(record["errors"], ensure_ascii=False)})
    json_write(args.output / "failures.json", [r for r in records if r["identity_cosine"] is None
               or r.get("hsemotion_cosine") is None
               or (not args.skip_openface and r.get("au_mae") is None)])
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["valid_identity_pairs"] or not summary["valid_hsemotion_pairs"] \
            or (not args.skip_openface and not summary["valid_au_pairs"]):
        raise RuntimeError("No valid pairs for a required metric; see failures and OpenFace logs")


if __name__ == "__main__":
    main()

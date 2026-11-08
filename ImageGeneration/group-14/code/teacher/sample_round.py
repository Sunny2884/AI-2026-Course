"""Generate exactly one teacher candidate per CSV row in an isolated round."""

import argparse
import csv
import hashlib
import json
import secrets
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import torch


def load_pipeline(model_path, device="cuda"):
    from diffusers import StableDiffusionInpaintPipeline
    from transformers import CLIPTextModel

    # Load CLIP directly: older Diffusers otherwise forwards an offload
    # argument that Transformers 4.57 passes to CLIP's constructor.
    text_encoder = CLIPTextModel.from_pretrained(
        str(model_path), subfolder="text_encoder", variant="fp16",
        dtype=torch.float16, use_safetensors=True, local_files_only=True,
    )
    return StableDiffusionInpaintPipeline.from_pretrained(
        str(model_path), text_encoder=text_encoder, variant="fp16",
        torch_dtype=torch.float16, use_safetensors=True, local_files_only=True,
        safety_checker=None,
    ).to(device)


def generate_batch(pipe, images, masks, seeds, args, device="cuda"):
    count = len(images)
    if not count or len(masks) != count or len(seeds) != count:
        raise ValueError("Each batch image requires exactly one mask and seed")
    generators = [torch.Generator(device).manual_seed(seed) for seed in seeds]
    with torch.inference_mode():
        results = pipe(
            prompt=[args.prompt] * count, negative_prompt=[args.negative_prompt] * count,
            image=images, mask_image=masks, strength=args.strength,
            guidance_scale=args.guidance, num_inference_steps=args.steps,
            num_images_per_prompt=1, generator=generators,
        ).images
    if len(results) != count:
        raise RuntimeError("Pipeline must return exactly one candidate per input image")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "detector", "data-root", "csv", "output-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--round", required=True, type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", required=True)
    args = parser.parse_args()
    if args.seed is None:
        args.seed = secrets.randbits(32)
    print("Sampling seed: {}".format(args.seed), flush=True)
    if args.round < 1 or args.resolution % 8 or not 0 < args.strength <= 1 or args.limit < 0 or args.batch_size < 1:
        raise ValueError("Invalid round, resolution, strength, limit or batch size")
    if not torch.cuda.is_available():
        raise RuntimeError("Run teacher sampling on a GPU worker")
    with args.csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[:args.limit]
    identifiers = [row.get("id") or Path(row["image_path"]).stem for row in rows]
    if not rows or len(set(identifiers)) != len(identifiers):
        raise ValueError("Expected nonempty CSV with unique image identifiers")
    if any(Path(i).name != i or not (args.data_root / row["image_path"]).is_file()
           for i, row in zip(identifiers, rows)):
        raise ValueError("Invalid ID or missing source image")
    output = args.output_root / ("round_{:03d}".format(args.round))
    # mkdir is exclusive: assigning the same round to two workers fails early.
    output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["csv_sha256"] = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    config["num_images"] = len(rows)
    config["role"] = "teacher pseudo-target generation; original image and text are teacher inputs only"
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    from insightface.model_zoo import get_model

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    detector = get_model(str(args.detector), providers=["CPUExecutionProvider"], sess_options=options)
    detector.prepare(ctx_id=-1, input_size=(320, 320), det_thresh=0.5)
    pipe = load_pipeline(args.model)
    print("Sampling batch size: {}".format(args.batch_size), flush=True)
    started = time.perf_counter()
    with (output / "manifest.jsonl").open("w") as manifest:
        for start in range(0, len(rows), args.batch_size):
            images, masks, seeds, records = [], [], [], []
            stop = min(start + args.batch_size, len(rows))
            for index in range(start, stop):
                row, image_id = rows[index], identifiers[index]
                source = args.data_root / row["image_path"]
                with Image.open(source) as opened:
                    image = opened.convert("RGB").resize((args.resolution, args.resolution), Image.Resampling.LANCZOS)
                boxes, _ = detector.detect(np.asarray(image)[:, :, ::-1].copy(), max_num=0)
                if len(boxes):
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    x1, y1, x2, y2 = boxes[int(np.argmax(areas)), :4]
                    mode = "largest_detected_face"
                else:
                    x1, y1, x2, y2 = (args.resolution / 4, args.resolution / 5,
                                      args.resolution * 3 / 4, args.resolution * 4 / 5)
                    mode = "center_fallback"
                px, py = (x2 - x1) * 0.38, (y2 - y1) * 0.55
                mask = Image.new("L", image.size, 0)
                ImageDraw.Draw(mask).ellipse((max(0, x1-px), max(0, y1-py),
                                              min(args.resolution, x2+px), min(args.resolution, y2+py)), fill=255)
                # Seeds do not depend on batch size or the image's batch position.
                seed_key = "{}:{}:{}".format(args.seed, args.round, row["image_path"])
                seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "big") % (2**63)
                images.append(image)
                masks.append(mask)
                seeds.append(seed)
                records.append({"id": image_id, "order": index+1, "round": args.round,
                                "source_path": str(source), "target_path": str(output / (image_id+".png")),
                                "seed": seed, "mask_mode": mode, "detected_faces": len(boxes)})
            try:
                results = generate_batch(pipe, images, masks, seeds, args)
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError("Sampling ran out of GPU memory; lower BATCH_SIZE in run_sample.sh") from error
            for result, record in zip(results, records):
                result.save(record["target_path"])
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()
            if stop // 10 != start // 10 or stop == len(rows):
                print("round {}: {}/{}; {:.2f}s/image".format(
                    args.round, stop, len(rows), (time.perf_counter()-started)/stop), flush=True)
    (output / "complete.json").write_text(json.dumps({"images": len(rows)}) + "\n")


if __name__ == "__main__":
    main()

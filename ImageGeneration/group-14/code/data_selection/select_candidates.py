"""Score teacher candidates per training source, distributed by source ID."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import torch

from scoring import SCORING_VERSION, WEIGHTS, composite_score, best_candidate


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "model-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--rounds", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--io-workers", type=int, default=4)
    args = parser.parse_args()
    if args.rounds < 1 or args.limit < 0 or args.batch_size < 1 or args.io_workers < 1:
        raise ValueError("Invalid round count, limit, batch size or I/O workers")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU selection requires CUDA")
        torch.cuda.set_device(local_rank)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = "cuda:{}".format(local_rank) if args.device == "cuda" else "cpu"
    code = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code/"evaluation"))
    from evaluate import FaceMetrics, crop_primary_face, load_image
    sys.path.insert(0, str(code/"data_selection"))
    from round_inventory import inspect_rounds
    root = args.project_root/"face_dataset"
    csv_path = root/"train.csv"
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        all_rows = list(csv.DictReader(handle))
    stems = [Path(row["image_path"]).stem for row in all_rows]
    if len(stems) != len(set(stems)) or not stems:
        raise ValueError("Invalid training CSV")
    selected_rows = all_rows[:args.limit] if args.limit else all_rows
    if world > len(selected_rows):
        raise ValueError("More processes than selected sources")
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(scoring_version=SCORING_VERSION, weights=WEIGHTS, rounds=args.rounds,
                  source_limit=args.limit, train_csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                  size=args.size, jpeg_quality=args.jpeg_quality, world_size=world, device=args.device,
                  source_count=len(selected_rows), model_root=str(args.model_root),
                  project_root=str(args.project_root),
                  condition_schema="8 expression probabilities, 2 raw VA, pitch/yaw/roll divided by 180")
    model_files = [args.model_root/"insightface"/name for name in
                   ("det_10g.onnx", "w600k_r50.onnx", "1k3d68.onnx")]
    model_files += [args.model_root/"hsemotion/enet_b0_8_va_mtl.pt",
                    args.model_root/"musiq/musiq_koniq_ckpt-e95806b9.pth"]
    config["model_hashes"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in model_files}
    config["versions"] = {name: importlib.metadata.version(name) for name in
                           ("torch", "pyiqa", "insightface", "onnxruntime", "timm")}
    round_root = args.project_root/"outputs/teacher_samples/train"
    # Persist each rank's config; merge validates equality and the exact ID set.
    shard = args.output/("rank_{:03d}".format(rank))
    if shard.exists() and not args.resume:
        raise FileExistsError("Use a fresh output or explicitly enable --resume")
    shard.mkdir(exist_ok=True)
    config_file = shard/"config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError("Resume configuration or model hash mismatch")
    write_json(config_file, config)
    write_json(shard/"execution.json", dict(batch_size=args.batch_size,io_workers=args.io_workers,
                                           rank=rank,device=device))
    if rank == 0:
        inventory = inspect_rounds(round_root, csv_path, stems, 1, args.rounds)
        write_json(args.output/"round_inventory.json", inventory)
        if inventory["unavailable_rounds"]:
            raise ValueError("Incomplete rounds: " + str(inventory["unavailable_rounds"]))
    rows = selected_rows[rank::world]
    completed = {}
    results = shard/"results"
    results.mkdir(exist_ok=True)
    for path in results.glob("*.json"):
        entry = json.loads(path.read_text())
        if entry["id"] != path.stem or entry["id"] in completed:
            raise ValueError("Invalid or duplicate completed source")
        completed[entry["id"]] = entry
    assigned = {Path(r["image_path"]).stem for r in rows}
    if set(completed)-assigned:
        raise ValueError("Resume data was produced by a different sharding layout")
    if set(completed)==assigned:
        write_json(shard/"complete.json", dict(rank=rank, world_size=world, sources=len(rows)))
        print("rank {}: all {} sources already complete".format(rank,len(rows)),flush=True)
        return

    faces = FaceMetrics(args.model_root/"insightface", "cpu", .5, 320)
    # FaceMetrics checks for CUDA at provider index 0. The device ID is set
    # explicitly for all sessions so each torchrun rank uses its own GPU.
    if args.device == "cuda":
        for model in (faces.detector, faces.identity, faces.landmark):
            model.session.set_providers(["CUDAExecutionProvider", "CPUExecutionProvider"],
                                        [{"device_id": local_rank}, {}])
            if model.session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("ONNX CUDA initialization fell back to CPU")
    expression = torch.load(model_files[-2], map_location="cpu", weights_only=False).to(device).eval()
    expression.requires_grad_(False)
    from torchvision import transforms
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                    transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def emotions(images):
        tensor = torch.stack([transform(Image.fromarray(image)) for image in images]).to(device)
        with torch.inference_mode():
            features = expression.forward_head(expression.forward_features(tensor), pre_logits=True)
            logits = expression.classifier(features)
        embedding, logits = features.float().cpu().numpy(), logits.float().cpu().numpy()
        if logits.shape != (len(images),10) or not np.isfinite(logits).all() or not np.isfinite(embedding).all():
            raise ValueError("Invalid HSEmotion output")
        norms = np.linalg.norm(embedding,axis=1,keepdims=True)
        if (norms <= 0).any():
            raise ValueError("Zero HSEmotion embedding")
        probabilities = torch.from_numpy(logits[:,:8]).softmax(dim=1).numpy()
        return embedding/norms, logits[:,8:], probabilities
    import pyiqa
    quality = pyiqa.create_metric("musiq", device=device,
                                 pretrained_model_path=str(model_files[-1])).eval()
    started = time.perf_counter()
    io_pool = ThreadPoolExecutor(max_workers=args.io_workers)
    print("rank {} candidate BATCH_SIZE={} I/O workers={}; face ONNX remains per-image".format(
        rank,args.batch_size,args.io_workers),flush=True)
    for index, row in enumerate(rows, 1):
        source = (root/row["image_path"]).resolve()
        image_id = source.stem
        if image_id in completed:
            continue
        original = load_image(source, args.size)
        face = faces.extract(original)
        candidates = []
        condition = None
        status = "source_face_missing"
        if face["face_count"] > 0:
            embeddings, va_values, probability_values = emotions([crop_primary_face(original, face["bbox"])])
            original_emotion, original_va, probabilities = embeddings[0], va_values[0], probability_values[0]
            condition = np.concatenate((probabilities, original_va, face["pose"]/180)).astype(float).tolist()
            if not np.isfinite(condition).all():
                raise ValueError("Invalid condition")
            status = "no_valid_candidate"
            for first in range(1,args.rounds+1,args.batch_size):
                numbers=list(range(first,min(first+args.batch_size,args.rounds+1)))
                paths=[round_root/("round_{:03d}".format(number))/(image_id+".png") for number in numbers]
                images=list(io_pool.map(lambda p:load_image(p,args.size,args.jpeg_quality),paths))
                candidate_faces=[faces.extract(image) for image in images]
                records=[dict(round=number,id=image_id,target_path=str(path),face_valid=0,quality_score=0.)
                         for number,path in zip(numbers,paths)]
                valid_indices=[i for i,f in enumerate(candidate_faces) if f["face_count"]>0]
                if valid_indices:
                    crops=[crop_primary_face(images[i],candidate_faces[i]["bbox"]) for i in valid_indices]
                    generated_embeddings,generated_vas,_=emotions(crops)
                    tensor=torch.stack([torch.from_numpy(images[i]).permute(2,0,1).float()/255 for i in valid_indices]).to(device)
                    with torch.inference_mode():
                        musiq_values=quality(tensor).reshape(-1).float().cpu().numpy()
                    if len(musiq_values)!=len(valid_indices):
                        raise ValueError("MUSIQ batch output count mismatch")
                    for position,i in enumerate(valid_indices):
                        record,candidate_face=records[i],candidate_faces[i]
                        delta=(candidate_face["pose"]-face["pose"]+180)%360-180
                        record.update(face_valid=1,
                                      identity_cosine=float(np.clip(np.dot(face["embedding"],candidate_face["embedding"]),-1,1)),
                                      pose_mae_deg=float(np.abs(delta).mean()),
                                      hsemotion_cosine=float(np.clip(np.dot(original_emotion,generated_embeddings[position]),-1,1)),
                                      hsemotion_va_mae=float(np.abs(original_va-generated_vas[position]).mean()),
                                      musiq_generated=float(musiq_values[position]))
                        record["quality_score"]=composite_score(record)
                candidates.extend(records)
        best = best_candidate(candidates)
        entry = dict(id=image_id, source_path=str(source), condition13=condition,
                     status="selected" if best else status, best=best, candidates=candidates)
        temporary = results/(image_id+".json.tmp")
        write_json(temporary,entry)
        os.replace(temporary, results/(image_id+".json"))
        if index%10 == 0 or index == len(rows):
            print("rank {} sources {}/{} elapsed {:.1f} min".format(rank,index,len(rows),(time.perf_counter()-started)/60),flush=True)
    io_pool.shutdown(wait=True)
    write_json(shard/"complete.json", dict(rank=rank, world_size=world, sources=len(rows)))


if __name__ == "__main__":
    main()

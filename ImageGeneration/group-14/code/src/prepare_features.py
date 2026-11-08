"""Cache CLIP hidden_states[-2] once and join the existing v2 auxiliary manifest."""

import argparse
import csv
import hashlib
import json
from datetime import timedelta
from pathlib import Path

from PIL import Image
import torch
from safetensors import safe_open, SafetensorError
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset

from model import FEATURE_SCHEMA, FEATURE_SHAPES


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    temporary.replace(path)


def source_rows(csv_path, data_root, pairs=None, limit=0):
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if pairs:
        selected = [json.loads(line) for line in Path(pairs).read_text().splitlines() if line.strip()]
        sources = [Path(row["source_path"]).resolve() for row in selected]
        if len(set(sources)) != len(sources) or set(sources) != {(data_root/row["image_path"]).resolve() for row in rows}:
            raise ValueError("Supervision must cover train.csv exactly once before LIMIT")
        rows = selected
    else:
        rows = [dict(id=row["id"], source_path=str((data_root/row["image_path"]).resolve())) for row in rows]
    if limit:
        rows = rows[:limit]
    ids = [row["id"] for row in rows]
    if not rows or len(ids) != len(set(ids)) or any(Path(value).name != value for value in ids):
        raise ValueError("Invalid source IDs")
    return rows


def clip_files(directory):
    directory = Path(directory)
    weight = next((directory/name for name in ("model.safetensors", "pytorch_model.bin") if (directory/name).is_file()), None)
    if weight is None or not (directory/"config.json").is_file():
        raise FileNotFoundError("Download h94/IP-Adapter models/image_encoder config.json and model.safetensors to "+str(directory))
    config = json.loads((directory/"config.json").read_text())
    if (config["hidden_size"], config["image_size"], config["patch_size"]) != (1280, 224, 14):
        raise ValueError("Expected the IP-Adapter Plus CLIP ViT-H/14 image encoder")
    return {"clip_config": directory/"config.json", "clip_weight": weight}


def cache_valid(path, metadata):
    if not Path(path).is_file():
        return False
    try:
        with safe_open(str(path), framework="pt") as handle:
            if handle.metadata() != metadata or set(handle.keys()) != {"clip"}:
                return False
        value = load_file(str(path))["clip"]
        return tuple(value.shape) == FEATURE_SHAPES["clip"] and bool(torch.isfinite(value).all())
    except (SafetensorError, OSError, ValueError, RuntimeError):
        return False

def reusable_clip(candidate, row, source_sha):
    """Read-only cache reuse: never unlink or recreate another job's cache links."""
    if not candidate or candidate.get('source_path') != row['source_path']:
        return None
    path=Path(candidate['clip_feature_path'])
    try:
        with safe_open(str(path),framework='pt') as handle:
            metadata=handle.metadata()
        if metadata.get('source_path')==row['source_path'] and metadata.get('source_sha256')==source_sha and cache_valid(path,metadata):
            return path.resolve()
    except (OSError,ValueError,RuntimeError,SafetensorError):
        pass
    return None


class SourceDataset(Dataset):
    def __init__(self, rows):
        from transformers import CLIPImageProcessor
        self.rows, self.processor = rows, CLIPImageProcessor()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["source_path"]) as image:
            pixels = self.processor(images=image.convert("RGB"), return_tensors="pt").pixel_values[0]
        return row["id"], pixels

def main():
    p=argparse.ArgumentParser()
    for name in ('csv','data-root','image-encoder','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reuse-cache',type=Path)
    p.add_argument('--pairs',type=Path)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    a=p.parse_args()
    from accelerate import Accelerator
    from accelerate.utils import broadcast_object_list,InitProcessGroupKwargs
    from filelock import FileLock
    acc=Accelerator(cpu=a.device=='cpu',kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=6))])
    torch.set_num_threads(4)
    files=clip_files(a.image_encoder)
    enc={k:dict(path=str(v),sha256=digest(v)) for k,v in files.items()}
    rows=source_rows(a.csv,a.data_root,a.pairs,a.limit)
    config=dict(schema=FEATURE_SCHEMA,encoders=enc,preprocessing='official_clip_image_processor_224_hidden_minus2_v1',
        csv_sha256=digest(a.csv),pairs_sha256=digest(a.pairs) if a.pairs else None,limit=a.limit,count=len(rows))
    config['fingerprint']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    fp=config['fingerprint'];a.output=a.output.resolve();status=[None];lock=None
    if acc.is_main_process:
        try:
            a.output.mkdir(parents=True,exist_ok=True);lock=FileLock(str(a.output/'.prepare.lock'),timeout=0);lock.acquire()
            if (a.output/'config.json').exists() and json.loads((a.output/'config.json').read_text())['fingerprint']!=fp:
                raise ValueError('Changed cache configuration; choose a fresh output')
            write_json(a.output/'config.json',config);(a.output/'clip_features').mkdir(exist_ok=True)
        except Exception as e: status[0]=str(e)
    if broadcast_object_list(status)[0]: raise RuntimeError(status[0])
    acc.wait_for_everyone()
    reuse={}
    if a.reuse_cache and (a.reuse_cache/'complete.json').exists():
        old=json.loads((a.reuse_cache/'config.json').read_text())
        if all(old['encoders'].get(k)==v for k,v in enc.items()):
            reuse={r['id']:r for r in [json.loads(l) for l in (a.reuse_cache/'conditions.jsonl').read_text().splitlines() if l.strip()]}
    pending=[];metadata={}
    for row in rows[acc.process_index::acc.num_processes]:
        m=dict(schema=FEATURE_SCHEMA,fingerprint=fp,source_path=row['source_path'],source_sha256=digest(row['source_path']))
        metadata[row['id']]=m;path=a.output/'clip_features'/(row['id']+'.safetensors')
        if not cache_valid(path,m):
            if reusable_clip(reuse.get(row['id']),row,m['source_sha256']) is not None:
                continue
            pending.append(row)
    print('rank {}: {} missing CLIP features; cached sources are reused'.format(acc.process_index,len(pending)),flush=True)
    if pending:
        from transformers import CLIPVisionModelWithProjection
        encoder=CLIPVisionModelWithProjection.from_pretrained(str(a.image_encoder),local_files_only=True).eval().requires_grad_(False).to(acc.device)
        for ids,pixels in DataLoader(SourceDataset(pending),batch_size=a.batch_size,num_workers=a.workers):
            with torch.inference_mode(): values=encoder(pixels.to(acc.device),output_hidden_states=True).hidden_states[-2]
            for i,key in enumerate(ids):
                value=values[i].half().cpu().contiguous()
                if tuple(value.shape)!=(257,1280) or not torch.isfinite(value).all(): raise ValueError('Bad CLIP feature')
                path=a.output/'clip_features'/(key+'.safetensors');tmp=path.with_suffix('.tmp')
                save_file({'clip':value},str(tmp),metadata=metadata[key]);tmp.replace(path)
    acc.wait_for_everyone()
    if acc.is_main_process:
        result=[]
        for row in rows:
            path=a.output/'clip_features'/(row['id']+'.safetensors')
            m=dict(schema=FEATURE_SCHEMA,fingerprint=fp,source_path=row['source_path'],source_sha256=digest(row['source_path']))
            if not cache_valid(path,m):
                path=reusable_clip(reuse.get(row['id']),row,m['source_sha256'])
                if path is None:raise ValueError('Missing or invalid cached CLIP: '+row['id'])
            with safe_open(str(path),framework='pt') as f: origin=f.metadata()['fingerprint']
            result.append({k:v for k,v in row.items() if k in ('id','source_path','target_path')} | dict(
                clip_feature_path=str(path.resolve()),feature_schema=FEATURE_SCHEMA,feature_fingerprint=fp,clip_origin_fingerprint=origin))
        tmp=a.output/'conditions.jsonl.tmp';tmp.write_text(''.join(json.dumps(r)+'\n' for r in result));tmp.replace(a.output/'conditions.jsonl')
        write_json(a.output/'complete.json',dict(count=len(result),fingerprint=fp,manifest_sha256=digest(a.output/'conditions.jsonl')))
        (a.output/'report.md').write_text('# CLIP-only cache\n\n{} sources, reusable raw CLIP [257,1280] fp16; no auxiliary encoders.\n'.format(len(result)))
    acc.wait_for_everyone()
    if lock:lock.release()
    acc.end_training()

if __name__=='__main__':main()



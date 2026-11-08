"""Validate all source shards and export a train-only selected-target manifest."""

import argparse
import csv
import json
import hashlib
from pathlib import Path
from collections import Counter

from scoring import best_candidate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--train-csv",type=Path,required=True)
    parser.add_argument("--data-root",type=Path,required=True)
    args=parser.parse_args()
    shards=sorted(args.output.glob("rank_*"))
    if not shards: raise ValueError("No rank outputs")
    configs=[json.loads((p/"config.json").read_text()) for p in shards]
    config=configs[0]
    if hashlib.sha256(args.train_csv.read_bytes()).hexdigest()!=config["train_csv_sha256"]:
        raise ValueError("Training CSV changed after scoring")
    if any(c!=config for c in configs) or len(shards)!=config["world_size"]:
        raise ValueError("Shard count/configuration mismatch")
    with args.train_csv.open(newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    if config["source_limit"]: rows=rows[:config["source_limit"]]
    sources={Path(r["image_path"]).stem:str((args.data_root/r["image_path"]).resolve()) for r in rows}
    results={}
    for rank,shard in enumerate(shards):
        done=json.loads((shard/"complete.json").read_text())
        if done["rank"]!=rank or done["world_size"]!=len(shards): raise ValueError("Incomplete/misnumbered rank")
        for path in (shard/"results").glob("*.json"):
            row=json.loads(path.read_text()); image_id=row["id"]
            if image_id not in sources or image_id in results or row["source_path"]!=sources[image_id]:
                raise ValueError("Unexpected, non-training or duplicate source")
            if row["condition13"] is not None:
                if len(row["candidates"])!=config["rounds"] or {r["round"] for r in row["candidates"]}!=set(range(1,config["rounds"]+1)):
                    raise ValueError("Missing candidate scores")
                if best_candidate(row["candidates"])!=row["best"]: raise ValueError("Incorrect best-of-64 selection")
            results[image_id]=row
    if set(results)!=set(sources): raise ValueError("Missing training sources")
    statuses=Counter(); rounds=Counter(); selected=[]
    with (args.output/"conditions.jsonl.tmp").open("w") as manifest, (args.output/"selected.csv.tmp").open("w",newline="") as pairs:
        writer=csv.DictWriter(pairs,fieldnames=["id","source_path","target_path","selected_round","quality_score"])
        writer.writeheader()
        for image_id in sources:
            row=results[image_id]; statuses[row["status"]]+=1
            if row["best"] is None: continue
            best=row["best"]
            if not Path(best["target_path"]).is_file(): raise FileNotFoundError(best["target_path"])
            item=dict(id=image_id,source_path=row["source_path"],target_path=best["target_path"],
                      selected_round=best["round"],quality_score=best["quality_score"])
            writer.writerow(item)
            manifest.write(json.dumps(dict(**item,condition13=row["condition13"]),ensure_ascii=False)+"\n")
            selected.append(item); rounds[best["round"]]+=1
    if not selected: raise RuntimeError("No valid selected training pairs")
    (args.output/"conditions.jsonl.tmp").replace(args.output/"conditions.jsonl")
    (args.output/"selected.csv.tmp").replace(args.output/"selected.csv")
    summary=dict(source_count=len(sources),selected=len(selected),statuses=dict(statuses),selected_round_counts=dict(rounds),
                 rounds=config["rounds"],mean_score=sum(r["quality_score"] for r in selected)/len(selected))
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (args.output/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n")
    report=["# 教师候选训练集构建", "", "源图 {}；每图候选 {}；选中 {}；排除 {}。".format(len(sources),config["rounds"],len(selected),len(sources)-len(selected)),
            "", "按固定综合分选择每张训练源图的最高分真实候选，相同分数优先较早轮次；只选检测有效候选。无源图主脸或64轮无有效候选时排除并保留原因。",
            "", "原图仅用于离线提取13维条件；学生训练读取conditions.jsonl中的条件和target_path，不读取原图RGB。所有源图经过原始train.csv白名单验证。",
            "", "当前综合分未包含照片/动漫风格分类器，MUSIQ并非真实性指标。候选排名不保证照片真实感，正式长训前应抽查选中图。",
            "", "平均选中综合分：{:.6f}。".format(summary["mean_score"]), "", "文件：selected.csv、conditions.jsonl、summary.json；每个rank/results保存64个候选分数和失败原因，支持复核和断点续跑。"]
    (args.output/"dataset_report.md").write_text("\n".join(report)+"\n")
    completion=dict(source_count=len(sources),selected=len(selected),rounds=config["rounds"],
                    manifest_sha256=hashlib.sha256((args.output/"conditions.jsonl").read_bytes()).hexdigest(),
                    full_training_split=config["source_limit"]==0)
    (args.output/"complete.json").write_text(json.dumps(completion,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__": main()

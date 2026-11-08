"""Generate and evaluate a chosen sequence of complete student checkpoints."""

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from check_cache import validate_cache

METRICS = {
    "quality_score": "up", "identity_cosine": "down", "pose_mae_deg": "down",
    "hsemotion_cosine": "up", "hsemotion_va_mae": "down", "au_mae": "down",
    "musiq_generated": "up", "musiq_delta": "up", "valid_identity_rate": "up",
}


def select_checkpoints(run, expression):
    if expression == "all":
        paths = sorted(p for p in run.glob("epoch_*") if (p/"complete.json").is_file())
    else:
        epochs = set()
        for item in expression.split(","):
            bounds = [int(value) for value in item.strip().split("-")]
            if len(bounds) == 1:
                bounds *= 2
            if len(bounds) != 2 or bounds[0] < 1 or bounds[1] < bounds[0]:
                raise ValueError("Use epochs such as 1-5 or 1,3,5")
            epochs.update(range(bounds[0], bounds[1]+1))
        paths = [run/("epoch_{:03d}".format(epoch)) for epoch in sorted(epochs)]
    if not paths:
        raise ValueError("No complete checkpoints selected")
    for path in paths:
        for name in ("complete.json", "config.json", "unet_lora.safetensors", "condition_adapter.safetensors",
                     "unet_ip_adapter.safetensors", "base_context.safetensors"):
            if not (path/name).is_file():
                raise FileNotFoundError(path/name)
    return paths


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def numeric(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def aggregate(summary, records, requested_count, composite_score):
    score = 0.
    for row in records:
        converted = {name: numeric(row.get(name)) for name in (
            "identity_cosine", "pose_mae_deg", "hsemotion_cosine", "hsemotion_va_mae", "musiq_generated")}
        converted["face_valid"] = all(value is not None for value in converted.values())
        score += composite_score(converted)
    values = {name: summary["metrics"].get(name, {}).get("mean") for name in METRICS}
    values.update(quality_score=score/requested_count,
                  valid_identity_rate=summary["valid_identity_pairs"]/requested_count,
                  requested=requested_count, generated=len(records),
                  valid_identity_pairs=summary["valid_identity_pairs"],
                  valid_hsemotion_pairs=summary["valid_hsemotion_pairs"],
                  valid_au_pairs=summary["valid_au_pairs"])
    return values


def report(output, results, split, writer, examples, requested):
    from baseline_report import baseline_row, baseline_data
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    header = ["epoch", "step", "complete_protocol", *METRICS, "requested", "generated", "valid_identity_pairs",
              "valid_hsemotion_pairs", "valid_au_pairs"]
    with (output/"metrics.csv").open("w", newline="") as handle:
        table = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        table.writeheader()
        table.writerows(results)
    save_json(output/"metrics.json", results)
    lines = ["# Checkpoint 评估", "", "数据划分：{}；固定源图数：{}；所有 checkpoint 使用同一 seed 和条件。".format(split, requested),
             "", "协议：{}；成功提取条件并生成 {}/{} 张。".format(
                 "完整（含 AU）" if results[-1]["complete_protocol"] else "非完整（显式跳过 AU）",
                 results[-1]["generated"], requested),
             "", "| Epoch | 综合分 ↑ | 身份余弦 ↓ | 姿态 MAE° ↓ | HSE 余弦 ↑ | VA MAE ↓ | AU MAE ↓ | MUSIQ ↑ | 有效身份配对 |",
             "|---|---|---|---|---|---|---|---|---|"]
    def fmt(value):
        return "N/A" if value is None else "{:.5f}".format(value)
    baseline, baseline_note = baseline_row(output, split)
    reference = baseline_data(output, split)
    lines.append(baseline)
    for row in results:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {}/{} |".format(
            row["epoch"], *[fmt(row[name]) for name in (
                "quality_score", "identity_cosine", "pose_mae_deg", "hsemotion_cosine",
                "hsemotion_va_mae", "au_mae", "musiq_generated")], row["valid_identity_pairs"], requested))
    lines += ["", baseline_note, "", "指标能正常计算不代表模型合格：身份余弦接近0也可能来自无关内容；姿态/AU仅在有效配对上计算，必须同时看覆盖率。",
              "HSEmotion在检测不到脸时仍可能对整图输出特征；MUSIQ也可能给清晰背景较高分。请结合以下固定样本检查是否生成了人脸、是否保持表情和姿态。", ""]
    if (output/"generation_audit.md").is_file():
        lines += ["## 生成异常核查", "", "![核查样例](comparison.png)", "",
                  "已逐图核对ID、源图路径、随机种子及checkpoint权重加载。详见 [生成核查记录](generation_audit.md)。", ""]
    eligible = [row for row in results if row["valid_identity_pairs"] and row["valid_hsemotion_pairs"]
                and (not row["complete_protocol"] or row["valid_au_pairs"])]
    if split == "val" and eligible:
        best = max(eligible, key=lambda row: (row["quality_score"], -row["epoch"]))
        save_json(output/"best_checkpoint.json", best)
        lines += ["", "验证集综合分最高：epoch {}，{}。".format(best["epoch"], best["checkpoint"])]
    else:
        lines += ["", "本次不自动选模（测试集，或没有具备有效指标的 checkpoint）。"]
    lines += ["", "## 统计口径与局限", "",
              "单项指标仅在可计算配对上取均值，分母见每轮 summary.json；缺失 AU 显示 N/A。",
              "综合分沿用 teacher_composite_v1：0.25 身份 + 0.20 姿态 + 0.20 表情 + 0.10 VA + 0.20 MUSIQ + 0.05 有效性。",
              "归一化分别为 clip((0.35-cos)/0.35)、exp(-pose/15)、clip((HSEcos-0.2)/0.8)、exp(-VA/0.5)、clip((MUSIQ-20)/60)，clip 范围 [0,1]。",
              "综合分逐真实候选计算；缺失生成、检测或必要特征的源图计 0，分母固定为请求源图数。AU 单独报告，不参与综合分。",
              "评估沿用现有主脸、256px、生成图 JPEG quality 75 协议；综合分为项目自定义指标，MUSIQ 不保证照片真实感。",
              "同一源图每 checkpoint 只生成一次，不按原图评分筛选随机种子。测试集只报告结果，不用于自动选模。", ""]
    figure, axes = plt.subplots(3, 3, figsize=(15, 11))
    for axis, (name, direction) in zip(axes.flat, METRICS.items()):
        x = [row["epoch"] for row in results]
        y = [float("nan") if row.get(name) is None else row[name] for row in results]
        axis.plot(x, y, "o-")
        reference_value = reference["means"].get(name) if reference else None
        if reference_value is not None:
            legend = "Baseline ({}, reference{})".format(reference["split"],
                " only" if reference["split"] != split else "")
            axis.axhline(reference_value, color="#b64a41", linestyle="--", label=legend)
            axis.legend(fontsize=8)
        axis.set(title=name+" ("+direction+")", xlabel="Epoch")
        axis.grid(alpha=.25)
        single, single_axis = plt.subplots(figsize=(7, 4))
        single_axis.plot(x, y, "o-")
        if reference_value is not None:
            single_axis.axhline(reference_value, color="#b64a41", linestyle="--", label=legend)
            single_axis.legend(fontsize=8)
        single_axis.set(title=name+" ("+direction+")", xlabel="Epoch")
        single_axis.grid(alpha=.25)
        single.tight_layout()
        single.savefig(output/(name+".png"), dpi=130)
        plt.close(single)
        for row in results[-1:]:
            if row.get(name) is not None:
                writer.add_scalar("evaluation/"+name, row[name], row["step"])
    figure.tight_layout()
    figure.savefig(output/"overview.png", dpi=140)
    plt.close(figure)
    lines += ["## 指标曲线", "", "![Overview](overview.png)", ""]
    for name in METRICS:
        lines += ["![{}]({}.png)".format(name, name), ""]
    if examples:
        tile, label = 192, 28
        def comparison(selected, filename):
            montage = Image.new("RGB", ((len(selected)+1)*tile, len(examples)*(tile+label)), "white")
            draw = ImageDraw.Draw(montage)
            for index, source in enumerate(examples):
                paths = [source["source_path"]]+[output/("epoch_{:03d}".format(row["epoch"]))/"generated"/(source["id"]+".png") for row in selected]
                for column, path in enumerate(paths):
                    with Image.open(path) as image:
                        montage.paste(image.convert("RGB").resize((tile, tile)), (column*tile, index*(tile+label)+label))
                    draw.text((column*tile+4, index*(tile+label)+4),
                              "source" if column == 0 else "epoch {}".format(selected[column-1]["epoch"]), fill="black")
            montage.save(output/filename)
            return montage
        montage = comparison(results, "comparison.png")
        import numpy as np
        writer.add_image("evaluation/comparison", np.asarray(montage), results[-1]["step"], dataformats="HWC")
        lines += ["## 固定样本对比", "", "按条件清单顺序取前{}张，不按得分挑选。".format(len(examples)),
                  "当前列为原图与各 checkpoint 推理图。训练集64轮候选和最佳监督图不能充当验证集同源样本，故本图未包含一次采样基线与64最佳监督图。",
                  "", "![Comparison](comparison.png)", ""]
        if len(results) > 5:
            lines += ["### 分组细节", "", "每组保留相同原图与相同五个样本，方便查看细节。", ""]
            for start in range(0, len(results), 5):
                selected = results[start:start+5]
                filename = "comparison_{:02d}_{:02d}.png".format(selected[0]["epoch"], selected[-1]["epoch"])
                comparison(selected, filename)
                lines += ["![Epoch {}-{}]({})".format(selected[0]["epoch"], selected[-1]["epoch"], filename), ""]
            chosen_epochs = {results[0]["epoch"], results[len(results)//2-1]["epoch"], results[-1]["epoch"]}
            if split == "val" and eligible:
                chosen_epochs.add(max(eligible, key=lambda row: (row["quality_score"], -row["epoch"]))["epoch"])
            selected = [row for row in results if row["epoch"] in chosen_epochs]
            comparison(selected, "comparison_key_epochs.png")
            lines += ["### 关键轮次", "", "固定比较首轮、中间轮与最后一轮；验证集额外加入综合分最高轮，重复轮次去重。", "",
                      "![Key epochs](comparison_key_epochs.png)", ""]
    (output/"report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "data-root", "model-root", "evaluation-code", "scoring-code", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--epochs", default="all")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="Metric device; generation requires GPU")
    parser.add_argument("--openface-bin", type=Path)
    parser.add_argument("--skip-openface", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--features-cache", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--guidance-scale", type=float, default=3.)
    parser.add_argument("--vae",type=Path,required=True)
    parser.add_argument("--sampler",choices=("ddim","dpmpp_2m_karras"),default="dpmpp_2m_karras")
    parser.add_argument("--ip-scale",type=float,default=0.8)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.threads) < 1 or args.limit < 0:
        raise ValueError("Invalid steps, batch size, threads or limit")
    checkpoints = select_checkpoints(args.run, args.epochs)
    if args.dry_run:
        print(json.dumps([str(path) for path in checkpoints], indent=2))
        return
    from torch.utils.tensorboard import SummaryWriter
    if not args.skip_openface and (not args.openface_bin or not args.openface_bin.is_file()):
        raise FileNotFoundError("OpenFace binary required; use --skip-openface only for an incomplete protocol")
    if not args.evaluation_code.is_file() or not args.scoring_code.is_file():
        raise FileNotFoundError("Evaluation and scoring code must be deployed on codelab")
    csv_path = args.data_root/(args.split+".csv")
    rows = read_rows(csv_path)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("Empty evaluation split")
    feature_config, all_sources = validate_cache(args.features_cache, csv_path, args.data_root)
    cached_by_id = {row["id"]: row for row in all_sources}
    missing = {row["id"] for row in rows}-set(cached_by_id)
    if missing:
        raise ValueError("Cached features missing requested IDs: "+str(sorted(missing)[:5]))
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/"config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    requested_csv = args.output/"requested.csv"
    write_rows(requested_csv, rows)
    here = Path(__file__).resolve().parent
    def run(command):
        print("Running:", command, flush=True)
        subprocess.run([sys.executable, *map(str, command)], check=True)
    (args.output/"conditions").mkdir()
    conditions = args.output/"conditions/conditions.jsonl"
    sources = [cached_by_id[row["id"]] for row in rows]
    conditions.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in sources))
    save_json(args.output/"conditions/config.json", {**feature_config,
              "cache_view": str(args.features_cache), "view_count": len(sources)})
    print("Reusing cached features for {} sources; no encoder extraction".format(len(sources)), flush=True)
    valid_ids = {row["id"] for row in sources}
    matched = [row for row in rows if row["id"] in valid_ids]
    if len(matched) != len(sources):
        raise ValueError("Condition IDs do not match evaluation CSV")
    matched_csv = args.output/"evaluated.csv"
    write_rows(matched_csv, matched)
    save_json(args.output/"coverage.json", dict(requested=len(rows), conditions=len(matched),
                                               excluded_ids=[row["id"] for row in rows if row["id"] not in valid_ids]))
    spec = importlib.util.spec_from_file_location("teacher_scoring", args.scoring_code)
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    results = []
    with SummaryWriter(str(args.output/"tensorboard")) as writer:
        for checkpoint in checkpoints:
            config = json.loads((checkpoint/"config.json").read_text())
            destination = args.output/checkpoint.name
            destination.mkdir()
            generated = destination/"generated"
            run([here/"infer.py", "--model", config.get("model", str(args.model_root/"stable-diffusion-inpainting")),
                 "--checkpoint", checkpoint, "--conditions", conditions, "--output", generated,
                 "--seed", args.seed, "--steps", args.steps, "--batch-size", args.batch_size, "--guidance-scale", args.guidance_scale,"--vae",args.vae,"--sampler",args.sampler,"--ip-scale",args.ip_scale])
            command = [args.evaluation_code, "--data-root", args.data_root, "--csv", matched_csv,
                       "--generated", generated, "--output", destination/"metrics",
                       "--insightface-root", args.model_root/"insightface",
                       "--musiq-weight", args.model_root/"musiq/musiq_koniq_ckpt-e95806b9.pth",
                       "--hsemotion-weight", args.model_root/"hsemotion/enet_b0_8_va_mtl.pt",
                       "--device", args.device, "--threads", args.threads]
            command += ["--skip-openface"] if args.skip_openface else ["--openface-bin", args.openface_bin]
            # Zero valid pairs is reported by the existing evaluator after writing diagnostics.
            completed = subprocess.run([sys.executable, *map(str, command)])
            summary_path = destination/"metrics/summary.json"
            if not summary_path.is_file():
                completed.check_returncode()
                raise RuntimeError("Evaluator did not produce a summary")
            summary = json.loads(summary_path.read_text())
            if completed.returncode and (summary["valid_identity_pairs"] and summary["valid_hsemotion_pairs"]
                                         and (args.skip_openface or summary["valid_au_pairs"])):
                completed.check_returncode()
            result = aggregate(summary, read_rows(destination/"metrics/per_image.csv"), len(rows), scoring.composite_score)
            result.update(epoch=config["epoch"], step=config["step"], checkpoint=str(checkpoint),
                          complete_protocol=not args.skip_openface)
            results.append(result)
            per_checkpoint = dict(**result, note="Metrics averaged on valid pairs; quality uses fixed requested denominator")
            save_json(destination/"result.json", per_checkpoint)
            (destination/"report.md").write_text("# 单 checkpoint 评估\n\n```json\n"+json.dumps(per_checkpoint, ensure_ascii=False, indent=2)+"\n```\n", encoding="utf-8")
            report(args.output, results, args.split, writer, sources[:5], len(rows))
            writer.flush()
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("Metrics:", args.output/"metrics.csv", "Report:", args.output/"report.md", flush=True)


if __name__ == "__main__":
    main()

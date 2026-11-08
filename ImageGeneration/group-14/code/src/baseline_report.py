"""Include measured baseline metrics with explicit split and protocol provenance."""

import json
import os
import csv
import math
import importlib.util
from pathlib import Path


def baseline_data(output, split):
    project = output.parents[2]
    configured = os.environ.get("BASELINE_EVALUATION")
    candidates = [Path(configured)] if configured else [
        project/"outputs/evaluation"/("baseline_"+split+"_cpu_hsemotion"),
        project/"outputs/evaluation/baseline_test_cpu_hsemotion"]
    directory = next((path for path in candidates if (path/"summary.json").is_file()), None)
    if directory is None:
        return None
    summary = json.loads((directory/"summary.json").read_text())
    config_path = directory/"config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    baseline_split = Path(config.get("csv", "unknown.csv")).stem
    means = {name: values.get("mean") for name, values in summary["metrics"].items()}
    means["valid_identity_rate"] = summary["valid_identity_pairs"]/summary["pairs"]
    quality_details = None
    per_image = directory/"per_image.csv"
    if per_image.is_file():
        run_config_path = output/"config.json"
        run_config = json.loads(run_config_path.read_text()) if run_config_path.is_file() else {}
        candidates = [Path(value) for value in (os.environ.get("SCORING_CODE"), run_config.get("scoring_code")) if value]
        candidates += [Path(__file__).resolve().parents[1]/"data_selection/scoring.py",
                       Path(__file__).resolve().parents[1]/"data_selection/scoring.py"]
        scoring_path = next((path for path in candidates if path.is_file()), None)
        if scoring_path is None:
            raise FileNotFoundError("Set SCORING_CODE to the deployed data_selection/scoring.py")
        spec = importlib.util.spec_from_file_location("baseline_scoring", scoring_path)
        scoring = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scoring)
        with per_image.open(newline="", encoding="utf-8-sig") as handle:
            records = list(csv.DictReader(handle))
        if len(records) != summary["pairs"] or len({row["id"] for row in records}) != len(records):
            raise ValueError("Baseline per-image count/IDs differ from its summary")
        total, valid = 0., 0
        for row in records:
            values = {}
            for name in ("identity_cosine", "pose_mae_deg", "hsemotion_cosine", "hsemotion_va_mae", "musiq_generated"):
                try:
                    value = float(row.get(name, ""))
                    values[name] = value if math.isfinite(value) else None
                except (ValueError, TypeError):
                    values[name] = None
            values["face_valid"] = all(value is not None for value in values.values())
            valid += int(values["face_valid"])
            total += scoring.composite_score(values)
        means["quality_score"] = total/len(records)
        quality_details = dict(version=scoring.SCORING_VERSION, mean=means["quality_score"],
                               pairs=len(records), valid=valid, failures_scored_zero=len(records)-valid,
                               source=str(per_image), denominator="all evaluated baseline source pairs")
    return dict(directory=directory, split=baseline_split, summary=summary, means=means,
                quality_details=quality_details)


def baseline_row(output, split):
    data = baseline_data(output, split)
    if data is None:
        return "| Baseline（尚无评测） | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |", \
               "Baseline尚无可读评测结果；设置BASELINE_EVALUATION为含summary.json的基线评估目录。"
    summary, directory, baseline_split = data["summary"], data["directory"], data["split"]
    # Only report measured values, never derive the composite from aggregate means.
    metrics = summary["metrics"]
    score = data["means"].get("quality_score")
    values = ["N/A" if score is None else "{:.5f}".format(score)]+["N/A" if metrics.get(key, {}).get("mean") is None
                       else "{:.5f}".format(metrics[key]["mean"]) for key in (
        "identity_cosine", "pose_mae_deg", "hsemotion_cosine", "hsemotion_va_mae", "au_mae", "musiq_generated")]
    count = summary["pairs"]
    row = "| Baseline（{}，{}张） | {} | {}/{} |".format(
        baseline_split, count, " | ".join(values), summary["valid_identity_pairs"], count)
    note = "Baseline来源：{}；".format(directory)
    if score is None:
        note += "没有完整逐图数据，综合分显示N/A，不从均值重建。"
    else:
        note += "综合分由per_image.csv逐图按teacher_composite_v1重算后平均，无效配对计0，固定分母{}。".format(count)
    if baseline_split != split:
        note += " **Baseline为{}，本次为{}，跨划分仅供参考，不能视作严格提升对照。**".format(baseline_split, split)
    else:
        note += " 同名划分仍需核对源图清单、分辨率和评测协议后才能作严格对照。"
    return row, note

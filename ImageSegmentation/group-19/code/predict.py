"""对 test 生成 COCO 掩码 JSON 和助教模板格式的掩码 CSV；只生成本地文件，不上传。"""
import argparse
import csv
import json
from collections import defaultdict

from pycocotools import mask as mask_utils

from rgbdseg.data import ComponentDataset
from rgbdseg.engine import collect_predictions
from rgbdseg.utils import load_model, make_loader, project_path, write_json


def export_submission(records, image_ids, template, output, csv_format="masks", partial=False):
    """将 COCO records 按 template 顺序导出到 output，返回图像数、实例数等摘要。

    image_ids 是本次推理的图像 ID；partial=True 仅导出这些图像的 preview 文件。
    csv_format 默认 masks，每格为该图的 JSON 掩码列表，与助教 CSV 模板结构一致。
    boxes 保留为可选的检测框字符串导出格式。
    masks 格式先去除面积为 0 的实例，再按置信度每图最多保留 100 个。
    无预测时两种格式都写 []；返回摘要包含去除的空掩码数量。
    """
    with template.open(encoding="utf-8-sig", newline="") as handle:
        template_ids = [int(row["image_id"]) for row in csv.DictReader(handle)]
    if len(template_ids) != len(set(template_ids)):
        raise ValueError("Duplicate image IDs in submission template.")
    if partial:
        selected_ids = set(image_ids)
        if not selected_ids.issubset(template_ids):
            raise ValueError("Test filenames do not match template IDs.")
        ordered_ids = [image_id for image_id in template_ids if image_id in selected_ids]
    else:
        if set(image_ids) != set(template_ids):
            raise ValueError("Test images and submission template do not have the same image IDs.")
        ordered_ids = template_ids
    by_image = defaultdict(list)
    empty_masks_removed = 0
    for record in records:
        if csv_format == "masks":
            rle = record["segmentation"]
            if rle["size"] != [1024, 1024] or not isinstance(rle["counts"], str):
                raise ValueError("Mask exports must be 1024x1024 COCO compressed RLE with string counts.")
            # 分类有分数时，掩码仍可能在 0.5 二值化后全为背景；不能作为实例提交。
            if int(mask_utils.area(rle)) == 0:
                empty_masks_removed += 1
                continue
        by_image[record["image_id"]].append(record)
    by_image = {image_id: sorted(by_image[image_id], key=lambda item: item["score"], reverse=True)[:100]
                for image_id in ordered_ids}
    exported_records = [record for image_id in ordered_ids for record in by_image[image_id]]
    output.mkdir(parents=True, exist_ok=True)
    name = "preview" if partial else "submission"
    write_json(output / f"{name}.json", exported_records, indent=None)
    with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "PredictionString"])
        for image_id in ordered_ids:
            instances = by_image[image_id]
            if csv_format == "masks":
                cell = json.dumps([{key: item[key] for key in ("category_id", "score", "segmentation")}
                                   for item in instances], separators=(",", ":"), allow_nan=False)
            else:
                values = []
                for item in instances:
                    x, y, width, height = item["bbox"]
                    values.extend([item["score"], x, y, x + width, y + height])
                cell = "[" + " ".join(f"{value:.6f}" for value in values) + "]"
            writer.writerow([image_id, cell])
    return {"images": len(ordered_ids), "instances": len(exported_records), "csv_format": csv_format,
            "empty_masks_removed": empty_masks_removed,
            "images_without_predictions": sum(not instances for instances in by_image.values()),
            "partial": partial, "basename": name}


def main():
    """读取 checkpoint，对测试图像推理并导出两种提交文件及来源记录；无返回值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--backbone-repo")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--depth-invalid-code", type=int, help="Explicit test invalid-depth code if test_images.json is missing")
    parser.add_argument("--template", default="submission.csv", help="Path relative to data root, or absolute")
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--csv-format", choices=["masks", "boxes"], default="masks",
                        help="masks matches the TA CSV template (JSON mask lists); boxes exports box strings")
    parser.add_argument("--limit", type=int, default=0, help="N>0 writes preview files, not a complete submission")
    parser.add_argument("--output", default="outputs/submission")
    args = parser.parse_args()
    if not 0.001 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0.001 and 1")
    model, config, device = load_model(args.checkpoint, args.device, args.backbone_repo)
    root = project_path(args.data_root or config["data_root"])
    dataset = ComponentDataset(root, "test", use_depth=config["use_depth"], limit=args.limit,
                               depth_invalid_code=args.depth_invalid_code)
    records = collect_predictions(model, make_loader(dataset, workers=args.workers),
                                  device, args.score_threshold)
    output = project_path(args.output)
    # Pathlib 允许右侧为绝对路径；相对模板路径则以数据根目录为起点。
    summary = export_submission(records, dataset.ids, root / args.template, output,
                                args.csv_format, partial=bool(args.limit))
    summary.update(checkpoint=str(project_path(args.checkpoint)), score_threshold=args.score_threshold,
                   depth_normalization="per-image relative depth" if config["use_depth"] else None,
                   test_invalid_codes=sorted({image["depth_invalid_code"] for image in dataset.images
                                              if "depth_invalid_code" in image}),
                   test_metadata_source="test_images.json" if (root / "test_images.json").is_file()
                                        else "filenames; explicit invalid-depth assumption")
    write_json(output / f"{summary['basename']}_info.json", summary)
    print(f"Saved {output}; images={summary['images']}; instances={summary['instances']}; "
          f"empty_masks_removed={summary['empty_masks_removed']}; "
          f"images_without_predictions={summary['images_without_predictions']}; partial={summary['partial']}")


if __name__ == "__main__":
    main()

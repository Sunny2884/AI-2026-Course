# 第19组 · 元器件 RGB-D 实例分割

本项目为每个元器件预测独立的分割掩码。最终方案将 RGB 与深度拼成四通道输入，使用 Swin-B 骨干的 Mask2Former。推理时结合整图与五个局部裁剪的预测，通过 mask NMS 合并重叠实例。下表展示在相同推理设置下，高倍率微调对验证集的影响。

| 2584 张验证图，COCO 分割指标 | 微调前 | 微调后 |
| --- | ---: | ---: |
| Mask AP | 0.7721 | 0.8001 |
| 小目标 Mask AP | 0.4892 | 0.5159 |

原始记录包含整图、五裁剪融合及独立验证子集的指标。[微调前](code/results/val_tiles5_baseline.json) · [微调后](code/results/val_tiles5_finetuned.json)

## 环境与输入

原实验使用 Python 3.11、PyTorch 2.7.1、CUDA 12.6 和四张 A100 40 GB。以下命令从本组目录开始执行。

```bash
cd code
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

将课程数据集置于 `code/dataset/`，其中包含 `rgb/`、`depth/`、`annotations/` 和测试提交模板 `submission_3.csv`。将云盘代码包中的 `weights_mask2former/` 复制到 `code/`，使 `weights_mask2former/swin_base_coco_instance/` 包含模型配置、预处理配置和预训练权重。若要复现 DINOv3 基线，还需将代码包中的 `external/dinov3/` 和 `weights/` 按原相对路径放入 `code/`。

## 训练与验证

先运行 30 轮训练，再从第一阶段的最佳检查点进行 6 轮高倍率微调。以下命令均在 `code/` 执行。

```bash
python train_m2f.py --config configs/config_pretrain.json --num-gpus 4
python train_m2f.py --config configs/config_finetune.json --num-gpus 4
python tools/run_tiled_pipeline.py \
  --run-root outputs/tiles5_run --checkpoint outputs/finetune/best.pth \
  --submission-dir outputs/submission_tiles5 --pilot-base probes/pilot \
  --allow-new-weights --phase val --max-gpus 3
```

验证汇总写入 `outputs/tiles5_run/val/summary.json`。生成测试集预测时，复用同一运行目录，将最后一条命令的 `--phase val` 改为 `--phase test`。原实验对 2112 张测试图的提交格式校验记录见[结果文件](code/submission/verification.json)。

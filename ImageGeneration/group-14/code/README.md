# CLIP-conditioned Face Anonymization

人脸匿名化课程项目的可移植代码。基于 **CLIP ViT-H/14 + IP-Adapter Plus + Realistic Vision V5.1 + UNet LoRA**，用筛选得到的匿名监督图学习“原图视觉特征→匿名图”的条件分布。支持两条复现路线：直接MSE VAE训练，以及原VAE训练后切换MSE VAE继续微调。

本目录提供特征提取、训练、评估和推理代码，GPU任务使用单机 `torchrun`。数据和权重按下方说明准备。课程提交路径为 `ImageGeneration/group-14/code/`，组级结果展示见上级 README。

## 方法

原图经过冻结CLIP图像编码器，提取倒数第二层 `[B,257,1280]` 隐藏状态并缓存。预训练Resampler转成 `[B,16,768]`，通过IP-Adapter的独立图像K/V注意力控制四通道SD1.5 UNet。推理从噪声生成整张图，不输入原图RGB/latent，不需要用户文本；原文本分支使用固定空条件。

训练 **Resampler、IP图像注意力K/V、UNet attention LoRA**；冻结 **CLIP、VAE、UNet原权重、生成固定空条件用的文本编码器**。10%训练样本把原始CLIP token置零，让同一网络学习空条件。CFG使用与训练相同的零特征，经同一Resampler生成无条件token。

$$
z_0=E_{VAE}(y),\quad z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon,
\quad L=\mathbb E\left[\|\epsilon_\theta(z_t,t,C(x))-\epsilon\|^2/(CHW)\right].
$$

这里 $x$ 是原图，$y$ 是选出的匿名监督图。训练使用去噪MSE，身份、表情、姿态和MUSIQ指标用于监督图筛选与评测。

$$
\hat\epsilon=\epsilon_{null}+s(\epsilon_{condition}-\epsilon_{null}).
$$

CFG系数 $s$ 与图像注意力的 `IP_SCALE` 不同。默认验证推理为 DPM++ 2M Karras、30步、CFG3、IP_SCALE0.8。

## 目录与路径

```text
face_anonymization_project_code_cjw/
  src/                    # 缓存、训练、初始化、推理、报告
  teacher/sample_round.py # 可选：教师一次采样
  data_selection/         # 64选1与共享综合分
  evaluation/evaluate.py # 身份/姿态/HSE/VA/AU/MUSIQ
  scripts/                # 所有运行入口；参数通过环境变量覆盖
  requirements.txt
  models/                 # 自行下载，gitignore
    realistic-vision-v5.1/
    sd-vae-ft-mse/
    ip_adapter/
      ip-adapter-plus_sd15.bin
      models/image_encoder/{config.json,model.safetensors}
    insightface/{det_10g.onnx,w600k_r50.onnx,1k3d68.onnx}
    musiq/musiq_koniq_ckpt-e95806b9.pth
    hsemotion/enet_b0_8_va_mtl.pt
    stable-diffusion-inpainting/ # 仅重新生成教师图需要
  workspace/              # 自行放数据，gitignore
    face_dataset/
      images/*.png
      train.csv
      val.csv
      test.csv
    data/selected64_v1/conditions.jsonl
    data/features_v4/{train,val,test}/
    outputs/teacher_samples/train/round_001 ... round_064/
    outputs/evaluation/
    checkpoints/
    tools/openface/{build/bin/FaceLandmarkImg,install/lib,install/lib64}/
```

通过环境变量设置数据、模型和OpenFace路径：

```bash
export PROJECT_ROOT=/absolute/path/to/workspace
export MODEL_ROOT=/absolute/path/to/models
export OPENFACE_ROOT=/absolute/path/to/openface
# 已自行激活环境则无需设置；设置后脚本用 source activate 激活
export CONDA_ENV=/absolute/path/to/conda/envs/cv
```

脚本能从任意工作目录运行，内部定位仓库并进入src。路径建议绝对路径。默认运行数据在 `workspace`，初始模型在 `models`。缓存清单会记录绝对路径及哈希，移动数据/模型后应重新构建清单/缓存，不要直接沿用指向旧机器的JSONL。`chmod 777`为本项目既有约定，脚本最后仅针对相应运行输出应用。

## 环境

Python3.10；安装匹配CUDA的PyTorch/torchvision后：

```bash
pip install -r requirements.txt
pip install --no-deps pyiqa==0.1.13
```

MUSIQ依赖的pyiqa 0.1.13为其他指标固定了Transformers 4.37.2，与本工程4.57冲突，因此显式安装MUSIQ运行依赖后单独以`--no-deps`安装pyiqa（本项目不调用那些其他指标）。

实验环境：torch2.7.1+cu126、torchvision0.22.1、diffusers0.35.1、transformers4.57.0、accelerate1.10.1、peft0.19.1、timm0.9.16。若使用其他PEFT版本，加载LoRA使用本项目严格默认adapter恢复器，以避免不必要的张量并行导入；恢复器适用于本项目使用的默认adapter权重格式；注意力计算使用PyTorch SDPA。GPU训练建议bf16。

OpenFace需要单独编译：参照 [OpenFace官方说明](https://github.com/TadasBaltrusaitis/OpenFace) 编译并放置FaceLandmarkImg及其所需模型/库。默认完整评估需要它。`SKIP_OPENFACE=1`可跳过AU，但报告会标明非完整协议。若只训练，`AUTO_EVAL=0`可不配置评估依赖。

## 权重下载

| 组件 | 来源 | 放置方式 |
|---|---|---|
| 四通道底座 | [Realistic Vision V5.1](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/tree/main) | models/realistic-vision-v5.1，Diffusers目录 |
| 官方MSE VAE | [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main) | config.json和diffusion_pytorch_model.safetensors |
| Plus适配器 | [ip-adapter-plus_sd15.bin](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.bin) | models/ip_adapter/ |
| CLIP图像编码器 | [h94/IP-Adapter models/image_encoder](https://huggingface.co/h94/IP-Adapter/tree/main/models/image_encoder) | config.json、model.safetensors；不是SD文本编码器 |
| InsightFace检测/身份/姿态 | [官方模型库](https://github.com/deepinsight/insightface/tree/master/model_zoo)，buffalo_l | 解压det_10g、w600k_r50、1k3d68到insightface/ |
| HSEmotion | [官方预训练模型](https://github.com/HSE-asavchenko/face-emotion-recognition/tree/main/models/affectnet_emotions) | enet_b0_8_va_mtl.pt |
| MUSIQ | [官方PyIQA权重](https://huggingface.co/chaofengc/IQA-PyTorch-Weights/tree/main) | musiq_koniq_ckpt-e95806b9.pth |
| 教师模型（可选） | [stable-diffusion-inpainting](https://huggingface.co/AI-ModelScope/stable-diffusion-inpainting) | 完整Diffusers，教师脚本要求fp16变体 |

底座至少保留以下文件：

```text
model_index.json
unet/{config.json,diffusion_pytorch_model.safetensors}
vae/{config.json,diffusion_pytorch_model.safetensors}
text_encoder/{config.json,model.safetensors}
scheduler/scheduler_config.json
tokenizer/{merges.txt,vocab.json,tokenizer_config.json,special_tokens_map.json}
```

底座自带VAE与官方MSE VAE是两套参数，路线B需要两套都保留。学生模型使用上面的Diffusers四通道UNet权重；根目录单文件.ckpt无需下载。代码不自动联网下载，文件齐全后离线运行。MSE VAE已核对的权重SHA256：`a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`；Plus为`1cb77fc0613369b66be1531cc452b823a4af7d87ee56956000a69fc39e3817ba`。

## 数据准备

原始课程FFHQ划分为train16000、val2000、test2000。数据不随代码发布，不重新随机拆分。CSV格式为 `image_path`（train必需）或 `id,image_path`（val/test必需），image_path相对face_dataset，例如 `images/55846.png`。val ID如val_00001；生成文件按CSV id命名。

训练配对JSONL每行一个对象：

```json
{"id":"00001","source_path":"/absolute/workspace/face_dataset/images/00001.png","target_path":"/absolute/workspace/outputs/teacher_samples/train/round_058/00001.png"}
```

清单需完整覆盖train.csv，每源图恰好一次；target_path指向选出的匿名监督图。输入原图只用于离线CLIP特征，Dataset读取监督图和特征。可用 `scripts/normalize_pairs.py --input <相对路径配对.jsonl> --output <conditions.jsonl> --data-root <face_dataset> --target-root <教师图根目录>` 将准备的相对路径配对转成当前机器绝对路径。

已有64最佳监督图时直接使用；无需重新采样。若从零构造监督：

```bash
# 先完成64轮，每次任务为全部训练原图生成一张候选；可将不同ROUND分配不同GPU
for ROUND in $(seq 1 64); do ROUND="$ROUND" bash scripts/sample_teacher.sh; done
# 检查所有轮次完整并评分、合并
GPUS=4 bash scripts/select_teacher.sh
```

教师可以使用原图、mask及文本，正式学生不使用这些输入。采样seed默认随机并写入manifest，因此重新采样不会逐像素复现已有监督；精确模型复现需要同一监督图及清单。教师推理脚本要求SD修复模型的fp16变体权重（unet、vae、text_encoder）。选择以256px、JPEG75及主脸提特征评分，依靠现有InsightFace/HSE/MUSIQ。64轮候选约102万张，需要较大存储。

## 路线A：直接MSE VAE训练（v5预训练起点）

```bash
# 例：4卡，每卡16，全局64；先准备CLIP缓存，再训练10轮，最后评估全部checkpoint
GPUS=4 BATCH_SIZE=16 GLOBAL_BATCH=64 SEED=20261031 bash scripts/train_direct.sh
```

路线A载入Realistic Vision和原始IP-Adapter Plus权重后，直接学习匿名图生成。冻结MSE VAE编码匿名监督图，训练Resampler/IP K/V/LoRA。默认每组学习率1e-4，LoRA rank32/alpha32/dropout0.05，AdamW weight decay0.01，warmup5%+cosine，clip1，bf16，512px。16000图/全局64每轮250次更新，10轮2500次。

## 路线B：原VAE先训练，再切MSE继续微调

```bash
# B1：重现v4阶段；底座原VAE，10轮；默认验证DDIM/CFG3/IP1
GPUS=4 BATCH_SIZE=16 GLOBAL_BATCH=64 SEED=1324400183 bash scripts/train_stage1.sh

# B2：从上一步完整epoch checkpoint初始化；请替换为本机真实目录
INIT_CHECKPOINT=/absolute/workspace/checkpoints/stage1_original_vae_<run_id>/epoch_009 \
GPUS=4 BATCH_SIZE=16 GLOBAL_BATCH=64 SEED=20261031 bash scripts/train_stage2.sh
```

B1完成后，从epoch 9开始B2。实验中B1的seed为1324400183，路线A及B2均为20261031。B2默认训练10轮，epoch从1重新计数；EPOCHS可调整第二阶段轮数。

INIT_CHECKPOINT只恢复LoRA、IP注意力、Resampler及固定空条件；**重新建立优化器、scheduler，global_step归零**。RESUME则恢复同一实验完整状态，与INIT_CHECKPOINT二选一；恢复时会检查VAE配置和权重哈希。checkpoint配置记录了底座绝对路径。迁移到另一台机器时，需要更新配置中的路径，并保留底座一致性检查。

## 特征缓存与并行

默认PREPARE_FEATURES=1：先验证train/val缓存，已有完整缓存只读使用；缺失时GPU提取CLIP并保存fp16 `[257,1280]`，再训练。调学习率、batch、VAE或LoRA不用重提特征。旧可复用CLIP通过源路径/哈希校验后直接引用，不反复删除创建链接。两个实验可共享同一缓存；第一次构建未完成时请先等它结束，文件锁会拒绝同时写同一缓存。已有缓存可设PREPARE_FEATURES=0。

单独准备：`SPLIT=train bash scripts/prepare.sh` 或 `SPLIT=val bash scripts/prepare.sh`。默认单节点多GPU，GLOBAL_BATCH须能整除BATCH_SIZE×GPUS。GPUS=8时可用每卡8；改用1卡时适当降低batch并自动累积。可用CUDA_VISIBLE_DEVICES分配两组独立GPU。

训练目录自动带上海时区时间戳和随机串。每epoch保存紧凑可训练状态、optimizer/scheduler/RNG以及独立推理权重；冻结大模型不重复保存。TensorBoard位于 `<运行目录>/tensorboard`，包括每步/区间/epoch损失、梯度、时间步分组、学习率、显存、吞吐和带噪目标x0预览。后者不是纯噪声生成结果。

## 验证、推理和选模

两条最终路线默认相同：MSE VAE、DPM++ 2M Karras、30步、CFG3、IP0.8、SEED20261026。训练末尾自动评估所有完整checkpoint。AUTO_EVAL=0可只训练，之后运行：

```bash
RUN_DIR=/absolute/workspace/checkpoints/<run_id> EPOCHS=1-10 bash scripts/evaluate.sh
# 离散轮次也支持EPOCHS=1,4,7
# test只汇报，不用于模型/超参数选择
RUN_DIR=/absolute/workspace/checkpoints/<run_id> EPOCHS=4 SPLIT=test bash scripts/evaluate.sh
```

独立test需先 `SPLIT=test bash scripts/prepare.sh`。跳过AU需显式SKIP_OPENFACE=1；没有有效必要指标时评估会报错并保留诊断。完整报告直接嵌入曲线与五图对比，保存metrics.csv/metrics.json/per_image.csv。训练数据清单为JSONL；评测CSV保留以兼容现有评测接口。BASELINE_EVALUATION可指定含config.json、summary.json和per_image.csv的基线评估目录；未提供时基线列显示N/A。

单次生成：

```bash
CHECKPOINT=/absolute/workspace/checkpoints/<run_id>/epoch_004 \
CONDITIONS=/absolute/workspace/data/features_v4/test/conditions.jsonl \
SAMPLER=dpmpp_2m_karras GUIDANCE_SCALE=3 IP_SCALE=0.8 SEED=20261026 bash scripts/infer.sh
```

测试时使用验证集选出的checkpoint和推理参数，每张原图生成一次。

## 指标与历史参考

| 结果 | 划分/数量 | 综合分 ↑ | 身份cos ↓ | 姿态MAE° ↓ | HSEcos ↑ | VA ↓ | AU ↓ | MUSIQ ↑ |
|---|---|---|---|---|---|---|---|---|
| 原修复baseline | test/2000 | 0.68860 | 0.05096 | 4.50925 | 0.49138 | 0.25588 | 0.49033 | 63.08109 |
| 路线A最佳epoch7 | val/2000 | 0.73782 | 0.05415 | 3.18187 | 0.57784 | 0.16069 | 0.39901 | 65.21270 |
| 路线B新增epoch4 | val/2000 | 0.74460 | 0.05138 | 3.11604 | 0.58235 | 0.15771 | 0.39354 | 65.86755 |

Baseline为test结果，A/B为val结果。A/B各有2000个有效身份配对，AU有效配对分别为1815、1801。

ArcFace单位向量余弦越低越匿名；姿态为pitch/yaw/roll最小环绕角误差绝对值均值；HSE为主脸特征余弦，越高越保持；VA为两维情绪回归MAE；AU为有效OpenFace强度MAE；MUSIQ为无参考图像质量。所有评测缩放256px，生成图JPEG75，主脸优先，单项只统计有效配对。HSE无检测脸时可能退回整图，MUSIQ反映清晰度等感知质量，皮肤纹理和妆感需要结合样例观察。

综合分使用data_selection/scoring.py，逐图归一化后加权：身份0.25、姿态0.20、HSE0.20、VA0.10、MUSIQ0.20、有效性0.05；无必要特征则计0，固定总源图分母，AU单独汇报。归一化为clip((.35-id)/.35)、exp(-pose/15)、clip((hse-.2)/.8)、exp(-VA/.5)、clip((MUSIQ-20)/60)，clip到[0,1]。

## 为什么两阶段可能更好（猜想）

1. **逐步适配**：第一阶段先在原底座熟悉的VAE latent上学习匿名监督映射；第二阶段不用同时从头学习匿名化和适应新VAE的编码统计。虽然两个VAE尺度相同，参数及编码分布并不完全相同。
2. **新学习率计划**：第一阶段后期cosine学习率很低；第二阶段重置优化器并重新warmup，给已经学到条件映射的模型一次针对新目标分布调整的机会。
3. **更多历史训练量**：第二阶段最佳权重继承第一阶段2250次更新，再新增1000次；路线A最佳为1750次。优势可能部分来自更多累计更新，而非两阶段策略本身。


MSE VAE已提升解码质量，但磨皮、妆感、年龄外观偏移仍存在；64选1监督本身也可能有这些偏差。后续希望从监督图入手改善自然皮肤纹理和外观保持。

## 参考实现

架构参考 [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)，扩散模型参考 [Diffusers](https://github.com/huggingface/diffusers)。第三方来源及许可证见THIRD_PARTY.md。数据和初始权重需自行获取；训练后权重也未打包。

## 打包课程 submission

先按验证集选定A epoch7或B第二阶段epoch4，对test每张图只生成一次；指定完整生成目录：

```bash
GENERATED=/absolute/workspace/outputs/inference_test \
SUBMISSION=/absolute/workspace/outputs/submission.csv bash scripts/submission.sh
```

脚本以正式test.csv的2000个ID严格校验，统一为256×256 JPEG quality75，再输出`id,image_b64`。生成的CSV可直接用于Kaggle提交。

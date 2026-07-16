# Jittor-CHIEF

用 **Jittor** 复现 CHIEF（Clinical Histopathology Imaging Evaluation Foundation）整张切片级模型，并提供与 PyTorch 参考实现逐项对齐的实验脚本。

> 论文：X. Wang, J. Zhao, E. Marostica, *et al.* (2024), “A pathology foundation model for cancer diagnosis and prognosis prediction,” **Nature 634**, 970–978. DOI: [10.1038/s41586-024-07894-z](https://doi.org/10.1038/s41586-024-07894-z)。
>
> 官方 PyTorch 代码：[hms-dbmi/CHIEF](https://github.com/hms-dbmi/CHIEF)。本仓库是独立整理的 Jittor 复现，不是作者官方发布。

## 1. 复现范围

论文完整 CHIEF 包含两级学习：tile 编码器的自监督预训练，以及 WSI 聚合器的弱监督预训练。原研究使用 60,530 张 WSI、19 个解剖部位和约 44 TB 图像，无法在单机课程作业中从零重训。

本仓库复现可公开、可验证且最适合 Jittor 对齐的部分：

- CHIEF WSI 级 gated-attention MIL 聚合器；
- 解剖部位文本嵌入到视觉空间的融合分支；
- 肿瘤来源等 slide-level 分类训练与测试；
- 官方 PyTorch `.pt/.pth` 特征和权重到框架中立格式的转换；
- PyTorch 与 Jittor 的前向结果、交叉熵和一次参数更新数值对齐；
- 小数据训练日志、性能日志、Loss 曲线、预测文件和注意力可视化；
- Linux CPU GitHub Actions 自动复现。

本仓库**不声称**在本地重现论文所有临床任务或论文表格中的最终数值。只有使用作者数据划分、预训练 tile 特征和官方模型权重，才能进行论文级结果比较。

## 2. 算法对应关系

一张 WSI 被切成 `N` 个 patch，每个 patch 由 tile encoder 编码为 768 维向量：

```text
WSI -> N 个 patch -> tile encoder -> H ∈ R^(N×768)
                                      |
                         Linear + ReLU + Dropout
                                      |
                      gated attention: tanh(.) ⊙ sigmoid(.)
                                      |
                           softmax over N patches
                                      |
                           加权求和得到 slide 表征
                                      |
                  + 可选的解剖部位文本向量投影
                                      |
                              线性分类器
```

门控注意力分数为：

```text
a_i = w_c^T [tanh(W_a h_i) ⊙ sigmoid(W_b h_i)]
α_i = exp(a_i) / Σ_j exp(a_j)
z   = Σ_i α_i h_i
```

若启用解剖信息，19 个部位的 768 维文本嵌入先投影到 512 维：

```text
z_fused = z + ReLU(W_text e_organ)
logits  = W_cls z_fused + b_cls
```

这种设计的核心是：训练标签只在 slide 级提供，模型通过注意力自行学习哪些 patch 对标签最重要；因此它属于弱监督多实例学习，而不是逐 patch 标注的监督学习。

## 3. 项目结构

```text
chief_jittor/model.py              Jittor CHIEF 主模型
chief_jittor/torch_reference.py    可测试的 PyTorch 参考实现
chief_jittor/data.py               可变长度 WSI bag 数据读取
chief_jittor/engine.py             两框架训练与验证循环
chief_jittor/weights.py            官方权重名称映射与 NPZ 转换
tools/align_torch_jittor.py        前向/损失/单步更新对齐
tools/compare_official_torch.py    与官方 CHIEF.py 的结构等价验证
tools/convert_pt_features.py       .pt patch 特征转 .npy
tools/convert_torch_checkpoint.py  官方 .pth 转通用 .npz
tools/create_common_init.py        生成两框架共同初始化
tools/benchmark.py                 推理性能记录
tools/plot_logs.py                 Loss 曲线
tools/visualize_attention.py       top-k patch 注意力图
scripts/make_toy_data.py           确定性合成 WSI 特征
scripts/run_hard_experiment.py     困难版三随机种子训练与测试入口
tools/summarize_hard_experiment.py 困难版均值/标准差与跨框架汇总
train.py / test.py                 统一训练与测试入口
.github/workflows/                 Linux Jittor 自动验证
```

## 4. 环境配置

### 推荐：Linux / Ubuntu 22.04

Jittor 在 Linux CPU/CUDA 上最稳定。建议 Python 3.10：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
sudo apt-get install -y g++ libomp-dev
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install torch jittor==1.3.11.0 pytest
python -m jittor.test.test_example
```

### Windows 说明

本机 Windows 11、Python 3.10/3.12、Jittor 1.3.11 的原生 CPU 初始化在 Jittor 自动生成 C++ 绑定时出现 MSVC `C2440`，错误发生在导入 Jittor 阶段，尚未进入本仓库模型。为了不把框架环境错误伪装成 CHIEF 结果，本仓库使用 Ubuntu CI 生成权威 Jittor 对齐日志。

Windows 可直接运行 PyTorch 参考链路；Jittor 推荐 WSL2、Docker 或 GitHub Actions。项目默认阻止 Windows 自动下载 1.85 GB CUDA 工具包；确需本机 CUDA 时，在 Python 启动前设置 `CHIEF_USE_CUDA=1`。

## 5. 数据准备

### 5.1 一键生成 smoke-test 数据

```bash
python scripts/make_toy_data.py --output data/toy --feature-dim 768 \
  --train 36 --val 12 --test 12 --seed 2026
```

每个 `.npy` 文件形状为 `[patch_count, 768]`。合成数据只用于验证代码和跨框架对齐，不代表真实病理任务。

### 5.2 转换官方 PyTorch patch 特征

```bash
python tools/convert_pt_features.py \
  --input /path/to/official_pt_features \
  --output data/features
```

数据清单 CSV：

```csv
case_id,label,organ,feature_path
TCGA-XX-0001,0,13,data/features/TCGA-XX-0001.npy
TCGA-XX-0002,1,13,data/features/TCGA-XX-0002.npy
```

- `case_id`：样本或切片 ID；
- `label`：从 0 开始的任务标签；
- `organ`：0–18 的解剖部位索引；关闭器官融合时可省略；
- `feature_path`：可省略，默认读取 `data_root/case_id.npy`。

数据划分必须按患者进行，不能让同一患者的不同切片跨越 train/val/test，否则会产生数据泄漏。

## 6. 官方权重转换

官方模型构造函数会直接读取 `model_weight/Text_emdding.pth`，且层名与本仓库略有不同。转换脚本处理 `attention_net -> fc/gated_attention` 和 `classifiers -> classifier`：

```bash
python tools/convert_torch_checkpoint.py \
  --input /path/to/official_chief.pth \
  --output model_weight/chief_official.npz
```

训练或微调时加载：

```bash
python train.py --config configs/tumor_origin_small.yaml \
  --backend jittor --device cuda \
  --init-weights model_weight/chief_official.npz
```

模型权重受官方发布条件约束，本仓库不重新分发。

## 7. PyTorch/Jittor 数值对齐

### 7.1 最严格的单步对齐

```bash
python tools/align_torch_jittor.py \
  --size small --patches 23 \
  --output logs/alignment_small.json
```

脚本使用同一份权重、同一输入、同一标签和 SGD 学习率，比较：

- `bag_logits`；
- `bag_prob`；
- 原始注意力与 softmax 注意力；
- WSI 表征与器官融合表征；
- 交叉熵；
- 一次更新后的 `fc.weight`。

默认通过标准：前向与损失最大绝对误差 `< 1e-5`，单步参数误差 `< 2e-5`。Linux CI 会在每次提交后自动运行。

此外，可验证清理后的 PyTorch 参考实现与官方 `models/CHIEF.py` 完全等价：

```bash
python tools/compare_official_torch.py \
  --official /path/to/hms-dbmi-CHIEF \
  --output logs/official_torch_equivalence.json
```

### 7.2 使用完全相同的初始化训练

```bash
python tools/create_common_init.py --config configs/toy.yaml \
  --output logs/toy/common_init.npz

python train.py --config configs/toy.yaml --backend torch \
  --init-weights logs/toy/common_init.npz
python train.py --config configs/toy.yaml --backend jittor \
  --init-weights logs/toy/common_init.npz
```

## 8. 训练、测试与演示

一键 smoke test：Windows PyTorch 运行 `powershell -ExecutionPolicy Bypass -File scripts/run_torch_smoke.ps1`；Linux Jittor 运行 `bash scripts/run_jittor_smoke.sh`。

```bash
# Jittor 训练
python train.py --config configs/toy.yaml --backend jittor --device cpu

# Jittor 测试，导出预测、指标和每张切片的注意力
python test.py --config configs/toy.yaml --backend jittor --device cpu

# 性能日志
python tools/benchmark.py --backend jittor --device cpu \
  --patches 512 --runs 30 --output logs/performance.jsonl

# Loss 曲线
python tools/plot_logs.py --output results/toy/loss_alignment.png

# 注意力 top-k 图
python tools/visualize_attention.py \
  --input results/toy/attention_jittor/test_0000.npz \
  --output results/toy/attention_top20_jittor.png
```

WSI 的 patch 数可能非常大。显存不足时可使用 `--max-patches 4096` 做确定性子采样；正式实验必须在 README/报告中注明采样上限。

### 8.1 录屏用的一键训练测试演示

建议在 Linux、WSL2 或 GitHub Actions 环境中运行。该脚本会依次完成：toy 数据生成、共同初始化、前向/损失/一步更新对齐、Jittor 训练、Jittor 测试、Loss 曲线和注意力图生成。

```bash
python scripts/run_jittor_demo.py --device cpu
```

如果数据和 `common_init.npz` 已经准备好，可缩短演示时间：

```bash
python scripts/run_jittor_demo.py --device cpu --skip-data
```

录屏时建议依次打开：

1. `chief_jittor/model.py`：讲解 `CHIEFJittor.execute()` 的输入、门控注意力、softmax、聚合和六个返回值；
2. `chief_jittor/engine.py`：指出 Jittor 使用 `optimizer.step(loss)` 完成反向传播与参数更新；
3. `logs/toy/train_jittor.jsonl`：展示逐轮训练/验证 Loss；
4. `checkpoints/toy/chief_jittor.pkl`：展示最佳验证 checkpoint；
5. `results/toy/metrics_jittor.json` 与 `predictions_jittor.csv`：展示总体指标和逐病例结果；
6. `results/toy/attention_jittor/` 与 `attention_top20_jittor.png`：展示 patch 级注意力证据；
7. [GitHub Actions run 29490678862](https://github.com/Yang-yang052/Jittor-CHIEF/actions/runs/29490678862)：证明 Linux 上真实完成训练、测试、三种子压力实验和结果回写。

课程汇报材料：

- [30 页算法与训练测试演示强化版 PPT](outputs/杨博涵-CHIEF-Jittor复现-算法与训练测试演示强化版.pptx)
- [约 25 分钟完整讲稿](outputs/杨博涵-CHIEF-Jittor复现-算法与训练测试演示强化版-25分钟讲稿.docx)
- [PPT 演讲者备注 Markdown](outputs/杨博涵-CHIEF-Jittor复现-算法与训练测试演示强化版-讲稿.md)

## 9. 当前可核验结果

PyTorch 本地环境：Windows 11，AMD Ryzen 9 8945HX，CPU PyTorch 2.13。Jittor 环境：GitHub Actions Ubuntu 22.04、Python 3.10.20、GCC 11.4、Jittor 1.3.11。两者使用相同随机种子 2026、相同 768 维模型、相同数据和相同初始化。

| 实验 | 结果 |
|---|---:|
| 单元测试 | 3 passed |
| 与官方 PyTorch `CHIEF.py` 等价性 | 4 个核心输出 max abs = 0 |
| PyTorch toy 训练 loss（epoch 1 → 8） | 1.0945 → 0.0090 |
| PyTorch toy 验证 loss（epoch 1 → 8） | 1.0698 → 0.3025 |
| PyTorch toy test accuracy | 1.0000 |
| PyTorch toy test macro-F1 | 1.0000 |
| PyTorch toy test macro-AUROC | 1.0000 |
| PyTorch CPU 推理，512 patches，30 次 | 4.889 ± 1.557 ms |
| Jittor toy 训练 loss（epoch 1 → 8） | 1.094550 → 0.009032 |
| Jittor toy 验证 loss（epoch 1 → 8） | 1.069769 → 0.302880 |
| Jittor toy test accuracy / macro-F1 / macro-AUROC | 1.0000 / 1.0000 / 1.0000 |
| 8 epochs 最大 train / val loss 差异 | 0.000644 / 0.000762 |
| 跨框架前向最大绝对误差 | 4.768 × 10⁻⁷ |
| 交叉熵绝对误差 | 0 |
| 单步更新后权重最大绝对误差 | 1.863 × 10⁻⁹ |
| Jittor CPU 推理，512 patches，30 次（Azure） | 0.086 ± 0.004 ms |
| Jittor 本机原生 Windows | 框架初始化失败，未伪造结果 |
| Jittor Linux CI | 对齐、8-epoch 训练、测试、性能与可视化全部通过 |

日志与图：

- `logs/toy/train_torch.jsonl`
- `logs/toy/train_jittor.jsonl`
- `logs/alignment_ci.json`
- `logs/performance.jsonl`
- `logs/performance_toy_jittor.jsonl`
- `results/toy/metrics_torch.json`
- `results/toy/metrics_jittor.json`
- `results/toy/loss_curve_torch.png`
- `results/toy/loss_alignment.png`
- `results/toy/attention_top20_torch.png`
- `results/toy/attention_top20_jittor.png`

toy 数据具有明确的类别原型，因此 1.0 指标只说明管线能学习和泛化到同分布合成数据，不能与 Nature 论文的临床结果横向比较。

### 9.1 困难版合成数据：三随机种子压力测试

为了避免把简单 toy 数据上的 `1.0` 误解为真实任务性能，仓库新增了一个更困难、但仍可完全复现的合成实验。它的目标不是模拟真实病理图像，而是检查在弱信号、噪声、bag 干扰和测试域偏移下，训练结论是否仍稳定、跨框架是否仍一致。

固定实验协议：

| 项目 | 设置 |
|---|---:|
| 随机种子 | 2026 / 2027 / 2028 |
| train / val / test | 180 / 60 / 90 bags（每个 seed） |
| 单个 bag 的 patch 数 | 64–160 |
| 特征维度 | 768 |
| 含类别信号的 patch | 12% |
| 类别信号强度 | 1.7 |
| bag 级干扰标准差 | 0.1 |
| 训练标签噪声 | 5% |
| 测试原型域偏移 | 0.2 |

PyTorch 与 Jittor 三随机种子测试结果：

| Seed | 框架 | Accuracy | Macro-F1 | Macro-AUROC |
|---:|---|---:|---:|---:|
| 2026 | PyTorch | 0.6333 | 0.6347 | 0.7598 |
| 2026 | Jittor | 0.6222 | 0.6228 | 0.7593 |
| 2027 | PyTorch | 0.4444 | 0.3623 | 0.8704 |
| 2027 | Jittor | 0.4444 | 0.3623 | 0.8704 |
| 2028 | PyTorch | 0.5000 | 0.5008 | 0.6815 |
| 2028 | Jittor | 0.5000 | 0.5008 | 0.6787 |
| **Mean ± SD** | **PyTorch** | **0.5259 ± 0.0971** | **0.4993 ± 0.1362** | **0.7706 ± 0.0949** |
| **Mean ± SD** | **Jittor** | **0.5222 ± 0.0909** | **0.4953 ± 0.1303** | **0.7694 ± 0.0962** |

三个 seed 上跨框架成对指标的最大绝对差为：Accuracy `0.0111`、Balanced Accuracy `0.0111`、Macro-F1 `0.0119`、Macro-AUROC `0.0028`。

Macro-AUROC 高于 Accuracy 并不矛盾：AUROC 衡量样本排序，不依赖固定阈值或 `argmax` 已经校准正确；模型可能已学会大致风险排序，但仍在类别决策边界上产生较多错误。

运行命令：

```bash
# Windows 可先运行 PyTorch 三种子实验
python scripts/run_hard_experiment.py --backend torch --device cpu

# Linux / WSL2 / GitHub Actions 运行 Jittor
python scripts/run_hard_experiment.py --backend jittor --device cpu

# 重新生成三种子均值、标准差、跨框架差值和曲线
python tools/summarize_hard_experiment.py --config configs/hard.yaml
```

主要证据文件：

- `configs/hard.yaml`
- `logs/hard/seed_*/resolved_config.yaml`
- `logs/hard/seed_*/common_init.npz`
- `logs/hard/seed_*/train_torch.jsonl`
- `logs/hard/seed_*/train_jittor.jsonl`
- `results/hard/seed_*/metrics_torch.json`
- `results/hard/seed_*/metrics_jittor.json`
- `results/hard/seed_*/predictions_torch.csv`
- `results/hard/seed_*/predictions_jittor.csv`
- `results/hard/summary.json`
- `results/hard/metrics_comparison.png`
- `results/hard/loss_alignment.png`

困难版 Jittor 三种子实验已加入 `.github/workflows/jittor-alignment.yml`。工作流复用相同生成参数和 `common_init.npz`，并把 Jittor 日志、指标、预测文件以及更新后的跨框架图自动回写仓库。

最终公开 CI 记录：[Jittor CHIEF alignment run 29490678862](https://github.com/Yang-yang052/Jittor-CHIEF/actions/runs/29490678862)。该运行的单元测试、前向/损失/单步更新对齐、Jittor 8-epoch 训练、测试、困难版三种子实验、性能测试、可视化和结果回写均为 `success`。PyTorch 与 Jittor 的性能数字来自不同机器，因此只能证明各自可运行，不能直接用于框架速度排名。

## 10. 真实任务复现检查表

- [ ] 获得官方权重及合法的数据访问权限；
- [ ] 使用论文相同 tile encoder 和 768 维 patch 特征；
- [ ] 严格复用患者级 train/val/test 划分；
- [ ] 固定 Jittor、Python、CUDA、cuDNN 和随机种子；
- [ ] 记录 PyTorch/Jittor 单步误差；
- [ ] 保存训练 JSONL、Loss 曲线、最佳 checkpoint；
- [ ] 报告 Accuracy、Balanced Accuracy、Macro-F1、AUROC；
- [ ] 报告 patch 数、推理时间和显存；
- [ ] 对错误病例和注意力区域做病理学解释；
- [ ] 明确区分 smoke test、外部验证与论文级复现。

## 11. License

CHIEF 官方代码声明 GPLv3 且限非商业学术用途。本移植按相同约束发布；使用官方权重和数据时还必须遵守其各自条款。详见 `LICENSE-NOTICE.md`。

## 12. Citation

```bibtex
@article{wang2024chief,
  title   = {A pathology foundation model for cancer diagnosis and prognosis prediction},
  author  = {Wang, Xiyue and Zhao, Junhan and Marostica, Eliana and others},
  journal = {Nature},
  volume  = {634},
  pages   = {970--978},
  year    = {2024},
  doi     = {10.1038/s41586-024-07894-z}
}
```

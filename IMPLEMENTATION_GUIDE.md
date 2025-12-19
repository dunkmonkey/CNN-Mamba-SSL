# Phase 0-3 实施完整指南

## 项目架构概览

本项目实现了双流（时频）Mamba + wav-KAN 的心音信号自监督表征学习框架，分为三个阶段递进实验。

### 核心技术栈
- **PyTorch Lightning**: 训练框架
- **Hydra**: 配置管理系统
- **Mamba SSM**: 状态空间模型（线性复杂度序列建模）
- **MoCo v2**: 动量对比学习框架
- **W&B**: 实验追踪
- **RTX 4090**: 24GB显存，优化大批次训练

---

## 阶段一：编码器验证（Phase 1）

### 目标
验证单流编码器的有效性，对比单向/双向 Mamba、时域/频域流。

### 消融实验

#### A1: 单向 vs 双向 Mamba
```bash
# 单向 Mamba（基线）
python scripts/train_pretrain.py experiment=ablation_a1 \
    model.mamba_config.bidirectional=false

# 双向 Mamba（实验组）
python scripts/train_pretrain.py experiment=ablation_a1 \
    model.mamba_config.bidirectional=true
```

**预期结果**: 双向 Mamba 应提升 2-5% top-1 准确率（参考 BiMamba 论文）

#### A2: 时域流 vs 频域流
```bash
# 时域流（CNN + Mamba）
python scripts/train_pretrain.py experiment=ablation_a2 model=time_stream

# 频域流（Vertical Strip + Mamba）
python scripts/train_pretrain.py experiment=ablation_a2 model=freq_stream
```

**预期结果**: 
- 时域流：更好捕捉 S1/S2 波形局部特征
- 频域流：更好提取杂音频谱特征

---

## 阶段二：MoCo 预训练优化（Phase 2）

### 目标
优化 MoCo 训练策略，验证增强强度和投影维度影响。

### 消融实验

#### B1: 增强策略强度
```bash
# 弱增强（仅噪声+轻度滤波）
python scripts/train_pretrain.py experiment=ablation_b1 augmentation=weak

# 强增强（噪声+滤波+失真+裁剪）
python scripts/train_pretrain.py experiment=ablation_b1 augmentation=strong

# 混合增强（时域强+频域弱）
python scripts/train_pretrain.py experiment=ablation_b1 augmentation=mixed
```

**预期结果**: 混合增强策略在下游任务上表现最佳（避免过度破坏病理特征）

#### B2: 投影头维度
```bash
# 64 维
python scripts/train_pretrain.py experiment=ablation_b2 \
    model.projection_config.output_dim=64

# 128 维（默认）
python scripts/train_pretrain.py experiment=ablation_b2 \
    model.projection_config.output_dim=128

# 256 维
python scripts/train_pretrain.py experiment=ablation_b2 \
    model.projection_config.output_dim=256

# 512 维
python scripts/train_pretrain.py experiment=ablation_b2 \
    model.projection_config.output_dim=512
```

**预期结果**: 128-256 维达到最佳平衡（参考 MoCo v2）

---

## 阶段三：双流融合与 wav-KAN（Phase 3）

### 目标
验证双流架构、GMU 融合和 wav-KAN 投影头的效果。

### 消融实验

#### C1: 融合策略
```bash
# GMU 门控融合（推荐）
python scripts/train_dual_pretrain.py experiment=ablation_c1 \
    model.fusion.mode=gmu

# 简单拼接（基线）
python scripts/train_dual_pretrain.py experiment=ablation_c1 \
    model.fusion.mode=concat

# 元素加法
python scripts/train_dual_pretrain.py experiment=ablation_c1 \
    model.fusion.mode=add
```

**预期结果**: GMU 自适应加权优于静态策略

#### C2: 投影头类型
```bash
# MLP 投影头（基线）
python scripts/train_dual_pretrain.py experiment=ablation_c2 \
    model.projection.type=mlp

# wav-KAN (morlet 小波)
python scripts/train_dual_pretrain.py experiment=ablation_c2 \
    model.projection.type=wavkan model.projection.wavelet_type=morlet

# wav-KAN (mexican_hat 小波)
python scripts/train_dual_pretrain.py experiment=ablation_c2 \
    model.projection.type=wavkan model.projection.wavelet_type=mexican_hat
```

**预期结果**: wav-KAN 在心音信号上优于 MLP（更好的时频局部性）

#### C3: 完整架构对比
```bash
# 单流时域（Phase 1 最优）
python scripts/train_dual_pretrain.py experiment=ablation_c3 model=time_stream

# 双流 + GMU
python scripts/train_dual_pretrain.py experiment=ablation_c3 model=dual_stream

# 双流 + GMU + wav-KAN（完整方案）
python scripts/train_dual_pretrain.py experiment=ablation_c3 model=dual_stream_wavkan
```

**预期结果**: 完整架构在 PhysioNet 2016 上达到 SOTA

---

## 关键配置文件索引

### 模型配置 (`configs/model/`)
| 文件 | 用途 |
|------|------|
| `time_stream.yaml` | 时域流（CNN + BiMamba） |
| `freq_stream.yaml` | 频域流（Vertical Strip + BiMamba） |
| `dual_stream.yaml` | 双流 + GMU + MLP |
| `dual_stream_wavkan.yaml` | 双流 + GMU + wav-KAN（完整方案） |
| `bimamba.yaml` | 双向 Mamba 配置 |
| `gmu.yaml` | GMU 融合配置 |

### 数据配置 (`configs/data/`)
| 文件 | 用途 |
|------|------|
| `physionet2016_v2.yaml` | 单模态（仅时域），2kHz |
| `physionet2016_dual.yaml` | 双模态（时域+频域），2kHz + 64 mels |

### 增强配置 (`configs/augmentation/`)
| 文件 | 用途 |
|------|------|
| `weak.yaml` | 弱增强（噪声 + 轻度低通滤波） |
| `strong.yaml` | 强增强（噪声 + 滤波 + 失真 + 裁剪） |
| `mixed.yaml` | 混合增强（时域强 + 频域 SpecAugment） |

### 实验配置 (`configs/experiment/`)
| 文件 | 消融点 | 阶段 |
|------|--------|------|
| `ablation_a1.yaml` | Uni-Mamba vs Bi-Mamba | Phase 1 |
| `ablation_a2.yaml` | Time-Only vs Freq-Only | Phase 1 |
| `ablation_b1.yaml` | Weak vs Strong vs Mixed Aug | Phase 2 |
| `ablation_b2.yaml` | Projection Dim 64/128/256/512 | Phase 2 |
| `ablation_c1.yaml` | GMU vs Concat vs Add | Phase 3 |
| `ablation_c2.yaml` | MLP vs wav-KAN | Phase 3 |
| `ablation_c3.yaml` | Single vs Dual+GMU vs Full | Phase 3 |

---

## 训练流程

### Phase 1 & 2: 单模态预训练
```bash
# 1. 使用默认配置
python scripts/train_pretrain.py

# 2. 使用实验配置
python scripts/train_pretrain.py experiment=ablation_a1 model.mamba_config.bidirectional=true

# 3. 自定义超参数
python scripts/train_pretrain.py \
    model=time_stream \
    data=physionet2016_v2 \
    trainer.max_epochs=300 \
    data.batch_size=256 \
    trainer.accumulate_grad_batches=4
```

### Phase 3: 双模态预训练
```bash
# 完整架构训练
python scripts/train_dual_pretrain.py \
    model=dual_stream_wavkan \
    data=physionet2016_dual \
    trainer.max_epochs=300 \
    data.batch_size=256
```

### 下游微调
```bash
# 线性探测（冻结编码器）
python scripts/train_finetune.py \
    pretrained_encoder=outputs/checkpoints/p3_dual_best.ckpt \
    trainer.freeze_encoder=true

# 全微调
python scripts/train_finetune.py \
    pretrained_encoder=outputs/checkpoints/p3_dual_best.ckpt \
    trainer.freeze_encoder=false
```

---

## RTX 4090 优化策略

### 显存优化
```yaml
# configs/trainer/pretrain.yaml
trainer:
  precision: 16-mixed  # 自动混合精度
  accumulate_grad_batches: 4  # 梯度累积
  max_epochs: 300
  accelerator: gpu
  devices: 1

data:
  batch_size: 256  # 24GB 显存可支持
  num_workers: 4
  pin_memory: true
  persistent_workers: true
```

### 预期性能
- **单流模型**: ~5-7 it/s @ batch_size=256
- **双流模型**: ~3-4 it/s @ batch_size=256
- **Phase 1-2 预训练**: ~6-8 小时（300 epochs）
- **Phase 3 预训练**: ~10-12 小时（300 epochs）

---

## W&B 实验追踪

### 配置
```yaml
# configs/logger/wandb.yaml
wandb:
  project: CNN-Mamba-SSL
  entity: your_wandb_username  # 修改为你的用户名
  save_dir: outputs/wandb
  log_model: true
  group: ${experiment_group}
  tags: ${tags}
```

### 关键指标
- `train_loss`: MoCo InfoNCE 损失
- `train_acc`: top-1 准确率（query 与 key 匹配）
- `val_loss`: 验证集损失
- `queue_ptr`: 队列指针（验证更新正常）

---

## 检查点命名规范

### 预训练检查点
```
p{phase}_{model}_{dataset}_{metric}_{timestamp}.ckpt
```

示例:
- `p1_time_stream_physionet2016_loss0.234_20240115.ckpt`
- `p3_dual_wavkan_physionet2016_loss0.189_20240120.ckpt`

### 加载检查点
```python
from src.utils.checkpoint_utils import load_encoder

encoder = load_encoder(
    'outputs/checkpoints/p3_dual_wavkan_best.ckpt',
    model_config=cfg.model
)
```

---

## 患者级数据划分

### 关键代码
```python
from src.data.utils.patient_split import PatientSplitter

splitter = PatientSplitter(
    dataset=full_dataset,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    stratify=True,  # 保持类别分布
    seed=42
)

train_idx, val_idx, test_idx = splitter.split()
```

### 重要性
防止数据泄漏：同一患者的所有音频段仅出现在一个集合中。

---

## 常见问题排查

### 1. CUDA Out of Memory
```bash
# 减少批次大小
python scripts/train_dual_pretrain.py data.batch_size=128

# 增加梯度累积
python scripts/train_dual_pretrain.py trainer.accumulate_grad_batches=8
```

### 2. Mamba 安装失败
```bash
# 确保 CUDA 版本匹配（需要 CUDA 11.8+）
pip install mamba-ssm causal-conv1d --no-cache-dir
```

### 3. W&B 认证
```bash
wandb login
# 输入 API key（从 wandb.ai/authorize 获取）
```

### 4. 数据集路径配置
```bash
# 方式 1：环境变量
export PHYSIONET_DATA_ROOT=/path/to/physionet2016

# 方式 2：命令行覆盖
python scripts/train_pretrain.py data.data_root=/path/to/physionet2016
```

---

## 下一步工作

1. **PhysioNet 2022 数据集集成**（Phase 4）
   - 更新 `physionet_dataset.py` 支持 2022 数据
   - 创建 `physionet2022_dual.yaml` 配置
   - 验证泛化能力

2. **超参数搜索**
   - 使用 Optuna 进行自动化调优
   - 重点调优：学习率、温度参数、队列大小

3. **可解释性分析**
   - Grad-CAM 可视化时频注意力
   - t-SNE 聚类验证表征质量

---

## 参考文献

- **MoCo v2**: Chen et al., "Improved Baselines with Momentum Contrastive Learning", arXiv 2020
- **Mamba**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", ICLR 2024
- **wav-KAN**: Bozorgasl & Chen, "Wav-KAN: Wavelet Kolmogorov-Arnold Networks", arXiv 2024
- **GMU**: Arevalo et al., "Gated Multimodal Units for Information Fusion", ICLR 2017

---

**最后更新**: 2024-01-15  
**项目维护**: dunkmonkey_20@github

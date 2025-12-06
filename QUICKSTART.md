# Quick Start Guide for CNN-Mamba-SSL

## 项目快速入门指南

本指南将帮助您快速了解和使用CNN-Mamba-SSL项目。

## 1. 项目概述

### 技术架构

```
输入: 心音信号 (1D时序数据, 4000Hz采样率)
    ↓
CNN前端: 提取局部形态特征
    - 3层1D卷积 (32→64→128通道)
    - 捕捉S1/S2心音波形、杂音纹理
    ↓
Mamba后端: 建模长程依赖
    - 4层状态空间模型 (d_model=128)
    - 分析心动周期节律、时序演变
    ↓
输出: 表征向量 (128维)
```

### 训练流程

```
阶段1: MoCo预训练 (无标签数据)
    - 数据增强生成双视图
    - 对比学习优化表征
    - 保存预训练编码器
    ↓
阶段2: 下游任务微调 (有标签数据)
    - 加载预训练权重
    - Linear Probing / Full Fine-tuning
    - 评估分类性能
```

## 2. 环境准备

### 系统要求

- Python 3.9+
- CUDA 11.7+ (GPU训练)
- 16GB+ GPU显存

### 安装依赖

```bash
# 1. 安装基础依赖
pip install -r requirements.txt

# 2. 安装Mamba (需要CUDA)
pip install mamba-ssm causal-conv1d

# 3. 验证安装
python -c "import torch; import mamba_ssm; print('✓ Installation successful!')"
```

### 常见安装问题

**问题1: Mamba安装失败**
```bash
# 解决方案: 从源码编译
pip install mamba-ssm --no-build-isolation
```

**问题2: CUDA版本不匹配**
```bash
# 检查CUDA版本
nvcc --version
# 安装对应版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 3. 数据准备

### 数据集结构

确保PhysioNet CinC 2016数据集按以下结构组织:

```
dataset/预训练大规模数据集/PhysioNetCinC Challenge 2016 Database/
├── challenge_2016_data_set/
│   ├── REFERENCE.csv         # 标注文件
│   ├── 0/                    # 正常样本 (~1000个.wav文件)
│   └── 1/                    # 异常样本 (~2200个.wav文件)
└── training-a/
    ├── a0001.hea            # 头文件
    ├── a0001.wav            # 音频文件
    └── ...
```

### 数据说明

- **采样率**: 多种(2000Hz为主),自动重采样至4000Hz
- **信号长度**: 变长(5-120秒),随机裁剪5秒片段
- **标签**: 二分类 (0=正常, 1=异常)
- **样本数**: ~3365个心音记录

## 4. 训练流程

### Step 1: MoCo预训练

**基础训练 (使用默认配置)**

```bash
python scripts/train_pretrain.py
```

**自定义配置训练**

```bash
python scripts/train_pretrain.py \
    experiment=exp001_baseline \
    trainer.trainer.max_epochs=200 \
    data.batch_size=64 \
    moco.queue_size=65536 \
    optimizer.lr=0.001
```

**训练输出**

- Checkpoint: `outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/`
- TensorBoard日志: `logs/pretrain/`
- 配置文件: `outputs/YYYY-MM-DD/HH-MM-SS/.hydra/`

**监控训练**

```bash
# 启动TensorBoard
tensorboard --logdir logs/

# 访问 http://localhost:6006
# 查看指标:
#   - train_loss: 对比损失
#   - train_acc: Top-1准确率
#   - val_loss: 验证损失
```

### Step 2: 导出预训练编码器

```bash
python scripts/export_encoder.py \
    --checkpoint outputs/2025-12-04/14-30-00/checkpoints/epoch=199-val_loss=0.1234.ckpt \
    --output pretrained_encoders/encoder_epoch199.pth
```

### Step 3: 下游任务微调

**准备微调数据集**

将少量有标签数据放入`dataset/微调数据集/`目录,格式与预训练数据相同。

**方式1: Linear Probing (快速)**

仅训练分类头,冻结编码器:

```bash
python scripts/train_finetune.py \
    pretrained_encoder=pretrained_encoders/encoder_epoch199.pth \
    trainer.freeze_encoder=true \
    data.data_root=dataset/微调数据集/ \
    trainer.trainer.max_epochs=50
```

**方式2: Full Fine-tuning (效果更好)**

端到端训练,编码器使用小学习率:

```bash
python scripts/train_finetune.py \
    pretrained_encoder=pretrained_encoders/encoder_epoch199.pth \
    trainer.freeze_encoder=false \
    trainer.encoder_lr_multiplier=0.01 \
    data.data_root=dataset/微调数据集/ \
    trainer.trainer.max_epochs=100
```

### Step 4: 模型评估

```bash
python scripts/evaluate.py \
    checkpoint=outputs/finetune/2025-12-04/15-00-00/checkpoints/best.ckpt \
    data.data_root=dataset/微调数据集/
```

**评估输出**

```
==============================================================
Detailed Classification Report:
==============================================================
              precision    recall  f1-score   support

      Normal       0.95      0.93      0.94       150
    Abnormal       0.96      0.97      0.97       250

    accuracy                           0.96       400
   macro avg       0.96      0.95      0.95       400
weighted avg       0.96      0.96      0.96       400

Confusion Matrix:
[[140  10]
 [  8 242]]

Predictions saved to: outputs/evaluation/predictions.npz
Confusion matrix saved to: outputs/evaluation/confusion_matrix.png
==============================================================
```

## 5. 配置调优

### 关键超参数

**MoCo超参数**
```yaml
moco:
  queue_size: 65536        # ↑ 更多负样本,效果更好但显存占用大
  momentum: 0.999          # ↑ 更平滑的更新
  temperature: 0.07        # ↓ 更区分的对比
```

**模型架构**
```yaml
model:
  cnn_config:
    out_channels: [32, 64, 128]  # 增加通道数提升容量
  mamba_config:
    n_layers: 4                   # 增加层数捕捉更长依赖
    d_state: 16                   # 增加状态维度
```

**数据增强**
```yaml
augmentation:
  gaussian_noise:
    prob: 0.5                     # 调整增强概率
  lowpass_filter:
    prob: 0.5                     # 最重要的增强!
```

### 显存优化技巧

```bash
# 1. 减小batch size
python scripts/train_pretrain.py data.batch_size=32

# 2. 减小队列大小
python scripts/train_pretrain.py moco.queue_size=16384

# 3. 使用混合精度训练 (已默认启用)
python scripts/train_pretrain.py trainer.trainer.precision=16-mixed

# 4. 使用梯度累积
python scripts/train_pretrain.py trainer.trainer.accumulate_grad_batches=2
```

## 6. 实验管理

### 消融实验

测试不同编码器架构:

```bash
python scripts/train_pretrain.py -m \
    experiment=exp002_ablation \
    model=cnn_only,mamba_only,cnn_mamba
```

### 超参数搜索

```bash
python scripts/train_pretrain.py -m \
    optimizer.lr=0.0001,0.001,0.01 \
    data.batch_size=32,64
```

### 实验对比

Hydra会自动在`outputs/multirun/`下为每个实验创建独立目录。

## 7. 故障排查

### 问题1: 数据加载慢

**原因**: 每次从.wav文件重新加载和重采样

**解决方案**: 启用缓存

```bash
python scripts/train_pretrain.py \
    data.use_cache=true \
    data.cache_dir=dataset/.cache/physionet2016
```

首次运行会预处理所有文件并缓存,后续训练直接从缓存加载。

### 问题2: 训练不收敛

**可能原因**:
1. 学习率过大 → 尝试 lr=0.0001
2. 增强太强 → 降低增强概率
3. Queue size太小 → 增加至65536

**调试步骤**:

```bash
# 1. 检查数据加载
python -c "
from src.data.datamodules.pretrain_datamodule import PretrainDataModule
dm = PretrainDataModule(data_root='dataset/预训练大规模数据集/PhysioNetCinC Challenge 2016 Database')
dm.setup('fit')
print(f'Train samples: {len(dm.train_dataset)}')
"

# 2. 小规模过拟合测试
python scripts/train_pretrain.py \
    trainer.trainer.max_epochs=10 \
    trainer.trainer.overfit_batches=10
```

### 问题3: 微调效果不好

**可能原因**:
1. 预训练不充分 → 延长预训练
2. 微调数据太少 → 使用数据增强
3. 学习率不合适 → 调整lr multiplier

**最佳实践**:
- 数据量<100: Linear Probing
- 数据量100-500: Full Fine-tuning (lr_multiplier=0.01)
- 数据量>500: Full Fine-tuning (lr_multiplier=0.1)

## 8. 进阶使用

### 自定义数据集

创建新的数据集类:

```python
# src/data/datasets/my_dataset.py
from src.data.datasets.physionet_dataset import PhysioNetPCGDataset

class MyDataset(PhysioNetPCGDataset):
    def _discover_files(self):
        # 自定义文件发现逻辑
        pass
```

### 自定义模型

修改编码器架构:

```yaml
# configs/model/my_model.yaml
_target_: src.models.encoders.hybrid_encoder.HybridEncoder

cnn_config:
  out_channels: [64, 128, 256, 512]  # 更深的CNN
  
mamba_config:
  n_layers: 6                         # 更多Mamba层
```

### 集成Wandb

```bash
pip install wandb

# 修改训练脚本,添加WandbLogger
from lightning.pytorch.loggers import WandbLogger
logger = WandbLogger(project="cnn-mamba-ssl", name="exp001")
```

## 9. 性能基准

### 预期性能 (PhysioNet 2016)

| 方法 | 预训练 | Accuracy | F1-Score | AUROC |
|------|--------|----------|----------|-------|
| CNN-only | ✗ | ~0.78 | ~0.76 | ~0.82 |
| CNN-Mamba | ✗ | ~0.82 | ~0.80 | ~0.86 |
| CNN-Mamba | ✓ (MoCo) | **~0.89** | **~0.87** | **~0.93** |

### 训练时间参考 (单GPU RTX 3090)

- 预训练 (200 epochs): ~12-24小时
- 微调 (100 epochs): ~2-4小时
- 评估: ~5分钟

## 10. 下一步

- ✅ 完成基础训练流程
- ⬜ 尝试不同数据增强策略
- ⬜ 进行消融实验分析各组件贡献
- ⬜ 在自己的数据集上评估
- ⬜ 调优超参数获得最佳性能
- ⬜ 撰写论文/报告

## 参考资源

- [MoCo论文](https://arxiv.org/abs/1911.05722)
- [Mamba论文](https://arxiv.org/abs/2312.00752)
- [PyTorch Lightning文档](https://lightning.ai/docs/pytorch/stable/)
- [Hydra文档](https://hydra.cc/docs/intro/)

---

**祝您实验顺利! 🚀**

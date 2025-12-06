# CNN-Mamba-SSL: 基于自监督对比学习与CNN-Mamba融合编码器的心音表征学习

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)

## 项目简介

本项目实现了一个创新的心音信号自监督表征学习框架,融合了卷积神经网络(CNN)的局部特征提取能力和Mamba状态空间模型的长程依赖建模能力,采用MoCo(Momentum Contrast)自监督学习范式进行预训练。

### 核心特性

- **混合编码器架构**: CNN前端提取短时形态特征 + Mamba后端捕捉长程节律模式
- **MoCo自监督学习**: 利用动量对比学习从大规模无标签数据中学习鲁棒表征
- **PyTorch Lightning框架**: 模块化设计,易于扩展和实验
- **Hydra配置管理**: 分层配置系统,支持灵活的实验管理
- **完整训练流程**: 预训练 → 微调 → 评估的端到端pipeline

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install mamba-ssm causal-conv1d  # 需要CUDA支持
```

### 2. MoCo预训练

```bash
python scripts/train_pretrain.py
```

### 3. 下游任务微调

```bash
python scripts/train_finetune.py \
    pretrained_encoder=path/to/encoder.pth \
    trainer.freeze_encoder=false
```

### 4. 模型评估

```bash
python scripts/evaluate.py checkpoint=path/to/checkpoint.ckpt
```

## 项目结构

详细的项目结构和使用说明请参考 `doc/` 目录下的文档。

## 技术架构

### CNN-Mamba混合编码器

1. **CNN前端**: 提取局部形态特征 (S1/S2心音、杂音)
2. **Mamba后端**: 捕捉长程节律模式 (心动周期、时序演变)
3. **投影头**: MLP映射到对比学习空间

### MoCo自监督学习

- 动态队列维护65536个负样本
- 动量更新 (τ=0.999) 保持key encoder稳定
- InfoNCE对比损失
- 6种心音增强策略

## 配置说明

使用Hydra配置系统,支持命令行覆盖:

```bash
python scripts/train_pretrain.py \
    model.cnn_config.out_channels=[64,128,256] \
    moco.queue_size=32768 \
    trainer.trainer.max_epochs=300
```

## 许可证

MIT License

## 致谢

基于 MoCo, Mamba, PyTorch Lightning, Hydra 等优秀开源项目

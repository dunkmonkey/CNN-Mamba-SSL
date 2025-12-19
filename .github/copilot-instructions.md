# CNN-Mamba-SSL Copilot 指南

## 项目概览
心音信号自监督表征学习框架，采用 MoCo 对比学习 + CNN-Mamba 混合编码器架构。

**技术栈**: PyTorch Lightning + Hydra + Mamba SSM
**数据流**: 心音(.wav) → 重采样4kHz → 5秒裁剪 → 双增强 → CNN前端 → Mamba后端 → 对比损失

## 核心架构
```
HybridEncoder (src/models/encoders/hybrid_encoder.py)
├── CNNEncoder: 局部形态特征 (S1/S2波形)
│   └── Conv1D × 3: [32→64→128] channels, stride=2
├── MambaEncoder: 长程节律模式
│   └── Mamba Blocks × 4: d_model=128, d_state=16
└── ProjectionHead: MLP映射到128维对比空间
```

## 关键文件
| 功能 | 文件 |
|------|------|
| MoCo预训练模块 | `src/models/moco_module.py` |
| 混合编码器 | `src/models/encoders/hybrid_encoder.py` |
| 数据增强（6种） | `src/data/datasets/augmentations.py` |
| PhysioNet数据集 | `src/data/datasets/physionet_dataset.py` |
| 主配置入口 | `configs/config.yaml` |

## 开发命令

```bash
# 1. 安装 (需要CUDA)
pip install -r requirements.txt
pip install mamba-ssm causal-conv1d

# 2. MoCo预训练
python scripts/train_pretrain.py

# 3. 下游微调 (支持线性探测/全微调)
python scripts/train_finetune.py pretrained_encoder=path/to/encoder.pth
python scripts/train_finetune.py trainer.freeze_encoder=true  # 线性探测

# 4. Hydra配置覆盖
python scripts/train_pretrain.py model.mamba_config.n_layers=6 moco.queue_size=32768
```

## Hydra 配置系统
采用分层组合模式，配置在 `configs/` 下分组：
- `model/`: cnn_mamba.yaml, cnn_only.yaml, mamba_only.yaml（消融实验）
- `trainer/`: pretrain.yaml, finetune.yaml
- `augmentation/`: pcg_augmentations.yaml（6种心音增强）
- `experiment/`: 预设实验组合

**配置引用语法**: `${output_dir}`, `${oc.env:PWD}`, `${now:%Y-%m-%d}`

## 代码规范

### Lightning Module 模式
所有训练模块继承 `L.LightningModule`，遵循：
- `__init__`: 调用 `self.save_hyperparameters(ignore=['encoder'])`
- `forward()`: 纯推理逻辑
- `training_step()`, `validation_step()`: 返回loss字典
- `configure_optimizers()`: 返回optimizer + scheduler

### 编码器接口
所有编码器必须实现 `get_output_dim() -> int` 方法，用于动态构建下游层。

### 数据增强
- 添加新增强：继承 `PCGAugmentation` 基类，实现 `apply()` 方法
- **禁止** Pitch Shift（会改变杂音病理特征）
- 低通滤波是最关键增强（模拟不同听诊器频响）

## MoCo 特定逻辑
- 动量更新在 `src/callbacks/momentum_update.py` 回调中执行
- Queue 大小=65536，动量τ=0.999，温度=0.07
- 预训练使用双增强视图：`PretrainDataModule.use_dual_augmentation=True`
- Key encoder 梯度冻结：`param.requires_grad = False`

## 常见扩展点

### 添加新编码器
1. 在 `src/models/encoders/` 创建文件
2. 实现 `get_output_dim()` 方法
3. 创建 `configs/model/xxx.yaml` 配置
4. 在 `hybrid_encoder.py` 的 `build_hybrid_encoder()` 中注册

### 消融实验
使用已有配置：`python scripts/train_pretrain.py model=cnn_only` 或 `model=mamba_only`

# CNN-Mamba-SSL Copilot 指南

## 项目概览
心音信号(PCG)自监督表征学习框架，采用 **MoCo v2 对比学习** + **CNN-Mamba 混合编码器**。

**技术栈**: PyTorch Lightning 2.0 + Hydra + Mamba SSM (需CUDA)  
**数据流**: 心音(.wav) → 重采样2-4kHz → 5秒裁剪 → 双增强视图 → 编码器 → InfoNCE对比损失

## 核心架构

### 单流编码器 (HybridEncoder)
```
CNN前端 [1→32→64→128] stride=2 → MambaEncoder (4层,d=128) → ProjectionHead (128-dim)
```
- **CNN**: 局部形态特征 (S1/S2波形边界)
- **Mamba**: 长程节律依赖 (心动周期模式)
- **BiMamba**: 双向扫描模式，融合方式: concat/gate/add

### 双流编码器 (DualStreamEncoder) - 进阶实验
```
时域流: CNN + BiMamba (625 tokens)  ─┬→ GMU门控融合 → Projection
频域流: Patch Embed + BiMamba (125 tokens) ─┘   (上采样对齐)
```

## 关键文件映射
| 功能 | 文件 |
|------|------|
| MoCo预训练 | `src/models/moco_module.py` (queue + momentum EMA) |
| 双模态MoCo | `src/models/dual_modal_moco_module.py` |
| 混合编码器 | `src/models/encoders/hybrid_encoder.py` |
| BiMamba实现 | `src/models/encoders/mamba_encoder.py` (BiMambaBlock) |
| GMU融合 | `src/models/fusion/gmu.py` |
| 增强策略 | `src/data/datasets/augmentations.py` (6种PCG增强) |
| 下游分类 | `src/models/classifier_module.py` |
| 主配置入口 | `configs/config.yaml` |
| 患者级划分 | `src/data/utils/patient_split.py` |

## 数据集配置

### 可用数据集
| 配置 | 数据集 | 采样率 | 用途 |
|------|--------|--------|------|
| `data=physionet2016` | PhysioNet 2016 | 4kHz | 基础预训练 |
| `data=physionet2016_v2` | PhysioNet 2016 | 2kHz | Phase 1 优化（patient-wise split） |
| `data=physionet2016_dual` | PhysioNet 2016 | 2kHz | 双流编码器（含Mel谱） |
| `data=finetune` | 微调数据集 | 4kHz | 二分类下游任务 |

### 切换数据集
```bash
python scripts/train_pretrain.py data=physionet2016_v2  # 使用2kHz版本
python scripts/train_pretrain.py data.batch_size=128 data.num_workers=8  # 覆盖参数
```

### 数据目录结构
```
dataset/
├── 预训练大规模数据集/PhysioNetCinC Challenge 2016 Database/
│   ├── training-a/ ~ training-f/  # 原始音频
│   └── challenge_2016_data_set/   # 标签文件
└── 微调数据集/  # 下游任务数据
```

## 开发命令

```bash
# 环境安装 (需要CUDA)
pip install -r requirements.txt
pip install mamba-ssm causal-conv1d

# MoCo预训练
python scripts/train_pretrain.py
python scripts/train_pretrain.py experiment=exp001_baseline  # 使用预设实验

# 下游微调
python scripts/train_finetune.py pretrained_encoder=path/to/encoder.pth
python scripts/train_finetune.py trainer.freeze_encoder=true  # 线性探测

# 消融实验 (切换编码器架构)
python scripts/train_pretrain.py model=cnn_only
python scripts/train_pretrain.py model=bimamba  # 双向Mamba
python scripts/train_pretrain.py model=dual_stream  # 时频双流

# Hydra配置覆盖
python scripts/train_pretrain.py model.mamba_config.n_layers=6 moco.queue_size=32768
```

## 验证与测试

```bash
# 验证文件结构和Python语法（无需依赖）
python scripts/verify_implementation.py

# 集成测试（测试模块导入、前向传播）
python scripts/test_integration.py

# 导出预训练编码器（从MoCo checkpoint提取encoder）
python scripts/export_encoder.py --checkpoint outputs/xxx/checkpoints/last.ckpt --output models/encoder.pth
```

## W&B 实验追踪

### 启用 Weights & Biases
```bash
# 安装
pip install wandb
wandb login

# 训练时启用
python scripts/train_pretrain.py logger=wandb
python scripts/train_pretrain.py logger=wandb logger.wandb.project=MyProject logger.wandb.tags=[ablation,bimamba]
```

### W&B 配置选项 (`configs/logger/wandb.yaml`)
- `project`: 项目名 (默认 CNN-Mamba-SSL)
- `group`: 实验分组 (如 phase1, phase2)
- `tags`: 标签列表
- `offline`: 离线模式 (无网络时设为 true)

## Hydra 配置系统
分层组合模式，通过 `defaults` 组装配置组：
- `model/`: cnn_mamba, bimamba, dual_stream, cnn_only, mamba_only
- `trainer/`: pretrain, finetune
- `augmentation/`: pcg_augmentations, mixed, weak, strong
- `experiment/`: ablation_a1~c3, exp001_baseline

**关键语法**: `${output_dir}`, `${oc.env:PWD}`, `${now:%Y-%m-%d}`  
**覆盖配置**: `experiment=ablation_a1 model.mamba_config.bidirectional=true`

## 代码模式规范

### Lightning Module
```python
class XXXModule(L.LightningModule):
    def __init__(self, encoder, ...):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])  # 必须排除模型对象
    
    def training_step(self, batch, batch_idx):
        return {"loss": loss}  # 返回字典
    
    def configure_optimizers(self):
        return {"optimizer": opt, "lr_scheduler": {...}}
```

### 编码器接口
**必须实现**: `get_output_dim() -> int` 用于动态构建分类头/投影头

### 数据增强 (PCGAugmentation)
```python
class MyAugmentation(PCGAugmentation):
    def apply(self, audio: np.ndarray) -> np.ndarray:
        return augmented_audio
```
- **禁止** Pitch Shift（改变杂音病理频率特征）
- **关键增强**: LowpassFilter（模拟不同听诊器频响）、GaussianNoise、TimeShift

## MoCo 实现细节
- **Queue**: 65536负样本，FIFO更新 (`_dequeue_and_enqueue`)
- **动量更新**: τ=0.999, θ_k = τ*θ_k + (1-τ)*θ_q (`_momentum_update_key_encoder`)
- **温度**: τ=0.07 (InfoNCE损失)
- **双增强**: 同一样本生成query/key视图 (`use_dual_augmentation=True`)
- **梯度隔离**: Key encoder参数 `requires_grad=False`

## 扩展指南

### 添加新编码器
1. 创建 `src/models/encoders/xxx_encoder.py`
2. 实现 `get_output_dim()` 方法
3. 创建 `configs/model/xxx.yaml` (使用 `_target_` 指定类路径)
4. 在 `hybrid_encoder.py` 的 `build_hybrid_encoder()` 中注册

### 添加新融合方式
1. 在 `src/models/fusion/` 创建模块
2. 在 `DualStreamEncoder` 中通过 `fusion_config.mode` 选择

### 消融实验配置
预设消融在 `configs/experiment/ablation_*.yaml`，使用变体覆盖：
```bash
python scripts/train_pretrain.py experiment=ablation_a1 model.mamba_config.bidirectional=false
python scripts/train_pretrain.py experiment=ablation_a1 model.mamba_config.bidirectional=true
```

## 完整工作流示例

```bash
# 1. 预训练 (MoCo)
python scripts/train_pretrain.py experiment=exp001_baseline data=physionet2016_v2

# 2. 导出编码器
python scripts/export_encoder.py --checkpoint outputs/2024-01-01/12-00-00/checkpoints/last.ckpt --output models/encoder.pth

# 3. 下游微调
python scripts/train_finetune.py pretrained_encoder=models/encoder.pth trainer.freeze_encoder=false

# 4. 评估
python scripts/evaluate.py checkpoint=outputs/finetune/checkpoints/best.ckpt
```

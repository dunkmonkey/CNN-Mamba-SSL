"""
Checkpoint utilities for standardized model saving and loading.

Naming convention: {phase}_{model}_{dataset}_{metric}_{timestamp}.ckpt
Example: p1_time_encoder_physionet2016_acc0.85_20251218.ckpt
"""

import os
import re
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union


# 检查点目录结构
CHECKPOINT_ROOT = "checkpoints"
PHASE_DIRS = {
    0: "phase0",
    1: "phase1", 
    2: "phase2",
    3: "phase3",
    4: "phase4"
}


def get_checkpoint_dir(phase: int, root_dir: Optional[str] = None) -> Path:
    """
    获取指定阶段的检查点目录。
    
    Args:
        phase: 阶段编号 (0-4)
        root_dir: 根目录，默认为当前工作目录
        
    Returns:
        检查点目录路径
    """
    root = Path(root_dir) if root_dir else Path.cwd()
    checkpoint_dir = root / CHECKPOINT_ROOT / PHASE_DIRS.get(phase, f"phase{phase}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def format_checkpoint_name(
    phase: int,
    model_name: str,
    dataset: str = "physionet2016",
    metric_name: str = "acc",
    metric_value: float = 0.0,
    timestamp: Optional[str] = None
) -> str:
    """
    生成标准化的检查点文件名。
    
    Args:
        phase: 阶段编号
        model_name: 模型名称 (如 time_encoder, freq_encoder, moco_time, dual_stream)
        dataset: 数据集名称
        metric_name: 指标名称 (如 acc, f1, loss)
        metric_value: 指标值
        timestamp: 时间戳，默认为当前时间
        
    Returns:
        格式化的文件名
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 清理模型名称，移除特殊字符
    model_name = re.sub(r'[^\w\-]', '_', model_name)
    
    return f"p{phase}_{model_name}_{dataset}_{metric_name}{metric_value:.4f}_{timestamp}.ckpt"


def parse_checkpoint_name(filename: str) -> Dict[str, Any]:
    """
    解析检查点文件名，提取元信息。
    
    Args:
        filename: 检查点文件名
        
    Returns:
        包含 phase, model_name, dataset, metric_name, metric_value, timestamp 的字典
    """
    pattern = r"p(\d+)_(.+)_(.+)_(\w+)([\d.]+)_(\d{8}_\d{6})\.ckpt"
    match = re.match(pattern, filename)
    
    if match:
        return {
            "phase": int(match.group(1)),
            "model_name": match.group(2),
            "dataset": match.group(3),
            "metric_name": match.group(4),
            "metric_value": float(match.group(5)),
            "timestamp": match.group(6)
        }
    return {}


def save_encoder(
    encoder: torch.nn.Module,
    phase: int,
    model_name: str,
    dataset: str = "physionet2016",
    metric_name: str = "acc",
    metric_value: float = 0.0,
    root_dir: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None
) -> Path:
    """
    保存编码器权重到标准化路径。
    
    Args:
        encoder: 要保存的编码器模型
        phase: 阶段编号
        model_name: 模型名称
        dataset: 数据集名称
        metric_name: 指标名称
        metric_value: 指标值
        root_dir: 根目录
        extra_info: 额外的元信息
        
    Returns:
        保存的文件路径
    """
    checkpoint_dir = get_checkpoint_dir(phase, root_dir)
    filename = format_checkpoint_name(
        phase, model_name, dataset, metric_name, metric_value
    )
    filepath = checkpoint_dir / filename
    
    # 构建保存内容
    save_dict = {
        "state_dict": encoder.state_dict(),
        "phase": phase,
        "model_name": model_name,
        "dataset": dataset,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "timestamp": datetime.now().isoformat()
    }
    
    if extra_info:
        save_dict["extra_info"] = extra_info
    
    torch.save(save_dict, filepath)
    print(f"[Checkpoint] Encoder saved to: {filepath}")
    
    # 同时保存一个 best 链接
    best_path = checkpoint_dir / f"p{phase}_{model_name}_best.ckpt"
    torch.save(save_dict, best_path)
    print(f"[Checkpoint] Best checkpoint updated: {best_path}")
    
    return filepath


def load_encoder(
    encoder: torch.nn.Module,
    checkpoint_path: Union[str, Path],
    strict: bool = True
) -> Dict[str, Any]:
    """
    加载编码器权重。
    
    Args:
        encoder: 目标编码器模型
        checkpoint_path: 检查点路径
        strict: 是否严格匹配权重键
        
    Returns:
        检查点元信息
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 加载权重
    if "state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["state_dict"], strict=strict)
    else:
        # 兼容直接保存的 state_dict
        encoder.load_state_dict(checkpoint, strict=strict)
    
    print(f"[Checkpoint] Encoder loaded from: {checkpoint_path}")
    
    # 返回元信息
    meta = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    return meta


def find_best_checkpoint(
    phase: int,
    model_name: str,
    root_dir: Optional[str] = None
) -> Optional[Path]:
    """
    查找指定阶段和模型的最佳检查点。
    
    Args:
        phase: 阶段编号
        model_name: 模型名称
        root_dir: 根目录
        
    Returns:
        最佳检查点路径，如果不存在则返回 None
    """
    checkpoint_dir = get_checkpoint_dir(phase, root_dir)
    best_path = checkpoint_dir / f"p{phase}_{model_name}_best.ckpt"
    
    if best_path.exists():
        return best_path
    
    # 尝试查找最新的检查点
    pattern = f"p{phase}_{model_name}_*.ckpt"
    checkpoints = list(checkpoint_dir.glob(pattern))
    
    if checkpoints:
        # 按时间戳排序，返回最新的
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]
    
    return None


def list_checkpoints(
    phase: Optional[int] = None,
    root_dir: Optional[str] = None
) -> Dict[int, list]:
    """
    列出所有检查点。
    
    Args:
        phase: 阶段编号，None 表示所有阶段
        root_dir: 根目录
        
    Returns:
        按阶段组织的检查点列表
    """
    root = Path(root_dir) if root_dir else Path.cwd()
    checkpoint_root = root / CHECKPOINT_ROOT
    
    result = {}
    
    phases_to_check = [phase] if phase is not None else range(5)
    
    for p in phases_to_check:
        phase_dir = checkpoint_root / PHASE_DIRS.get(p, f"phase{p}")
        if phase_dir.exists():
            checkpoints = list(phase_dir.glob("*.ckpt"))
            if checkpoints:
                result[p] = [
                    {
                        "path": str(ckpt),
                        "name": ckpt.name,
                        **parse_checkpoint_name(ckpt.name)
                    }
                    for ckpt in checkpoints
                ]
    
    return result

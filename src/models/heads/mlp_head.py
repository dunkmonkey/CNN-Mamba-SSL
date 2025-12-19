"""
MLP Projection Head for contrastive learning.

Standard projection head used in MoCo, SimCLR, etc.
Maps encoder outputs to a lower-dimensional embedding space
where contrastive loss is computed.
"""

import torch
import torch.nn as nn
from typing import Optional


class MLPProjectionHead(nn.Module):
    """
    MLP 投影头，用于对比学习。
    
    将编码器输出映射到低维嵌入空间，在该空间中计算对比损失。
    标准结构：Linear -> ReLU -> Linear
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 128,
        num_layers: int = 2,
        use_batchnorm: bool = False,
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: 输入特征维度（编码器输出维度）
            hidden_dim: 隐藏层维度
            output_dim: 输出嵌入维度
            num_layers: 层数（必须 >= 2）
            use_batchnorm: 是否使用 BatchNorm
            dropout: Dropout 概率
        """
        super().__init__()
        
        assert num_layers >= 2, "MLPProjectionHead must have at least 2 layers"
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        
        # 第一层
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # 中间层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # 输出层（无激活函数）
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入特征 (batch_size, input_dim)
            
        Returns:
            嵌入向量 (batch_size, output_dim)
        """
        return self.mlp(x)
    
    def get_output_dim(self) -> int:
        """获取输出维度。"""
        return self.output_dim


class SimCLRProjectionHead(MLPProjectionHead):
    """
    SimCLR 风格的投影头。
    
    与标准 MLP 相同，但在输出上添加 L2 归一化。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 128,
        num_layers: int = 2,
        use_batchnorm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            use_batchnorm=use_batchnorm,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，带 L2 归一化。
        
        Args:
            x: 输入特征 (batch_size, input_dim)
            
        Returns:
            归一化的嵌入向量 (batch_size, output_dim)
        """
        z = self.mlp(x)
        return nn.functional.normalize(z, dim=1)

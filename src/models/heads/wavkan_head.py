"""
Wavelet-based Kolmogorov-Arnold Network (wav-KAN) Projection Head.

Implements a projection head using learnable wavelet activation functions
instead of fixed activation functions (ReLU, etc.).

Reference:
- KAN: Kolmogorov-Arnold Networks (Liu et al., 2024)
- Wav-KAN: Wavelet Kolmogorov-Arnold Networks (Bozorgasl & Chen, 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Literal, Optional


class WaveletTransform(nn.Module):
    """
    可学习的小波变换层。
    
    使用小波基函数作为激活函数，参数（尺度、平移、权重）可学习。
    
    支持的小波类型：
    - mexican_hat: Mexican Hat (Ricker) 小波，适合边缘检测和瞬态特征
    - morlet: Morlet 小波，适合时频分析
    - dog: Difference of Gaussians，梯度特征
    - shannon: Shannon 小波，频带分离
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        wavelet_type: Literal["mexican_hat", "morlet", "dog", "shannon"] = "morlet"
    ):
        """
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            wavelet_type: 小波类型
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
        
        # 可学习参数
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # 初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.scale, 0.5, 2.0)
        nn.init.uniform_(self.translation, -1.0, 1.0)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)
    
    def mexican_hat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mexican Hat (Ricker) 小波。
        
        ψ(t) = (2/√3π^0.25) * (1 - t²) * exp(-t²/2)
        """
        return (1 - x ** 2) * torch.exp(-x ** 2 / 2)
    
    def morlet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Morlet 小波（实部）。
        
        ψ(t) = exp(-t²/2) * cos(5t)
        """
        return torch.exp(-x ** 2 / 2) * torch.cos(5 * x)
    
    def dog(self, x: torch.Tensor) -> torch.Tensor:
        """
        Difference of Gaussians (DOG) 小波。
        """
        return x * torch.exp(-x ** 2 / 2)
    
    def shannon(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shannon 小波（简化版）。
        
        sinc 函数近似
        """
        # 避免除零
        x_safe = torch.where(x == 0, torch.ones_like(x) * 1e-8, x)
        return torch.sin(math.pi * x_safe) / (math.pi * x_safe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入 (batch_size, in_features)
            
        Returns:
            输出 (batch_size, out_features)
        """
        batch_size = x.size(0)
        
        # 扩展维度以进行广播
        # x: (B, 1, in) -> 与 (out, in) 运算
        x_expanded = x.unsqueeze(1)  # (B, 1, in)
        
        # 缩放和平移
        x_scaled = (x_expanded - self.translation) / (self.scale + 1e-8)  # (B, out, in)
        
        # 应用小波变换
        if self.wavelet_type == "mexican_hat":
            wavelet_out = self.mexican_hat(x_scaled)
        elif self.wavelet_type == "morlet":
            wavelet_out = self.morlet(x_scaled)
        elif self.wavelet_type == "dog":
            wavelet_out = self.dog(x_scaled)
        elif self.wavelet_type == "shannon":
            wavelet_out = self.shannon(x_scaled)
        else:
            raise ValueError(f"Unknown wavelet type: {self.wavelet_type}")
        
        # 加权求和
        output = (wavelet_out * self.weight).sum(dim=-1)  # (B, out)
        
        # Batch normalization
        output = self.bn(output)
        
        return output


class KANLinear(nn.Module):
    """
    KAN 线性层。
    
    结合小波变换和线性变换。
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        wavelet_type: str = "morlet",
        use_base_linear: bool = True
    ):
        """
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            wavelet_type: 小波类型
            use_base_linear: 是否包含基础线性变换
        """
        super().__init__()
        
        self.wavelet = WaveletTransform(in_features, out_features, wavelet_type)
        
        self.use_base_linear = use_base_linear
        if use_base_linear:
            self.base_linear = nn.Linear(in_features, out_features)
            self.combine_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入 (batch_size, in_features)
            
        Returns:
            输出 (batch_size, out_features)
        """
        wavelet_out = self.wavelet(x)
        
        if self.use_base_linear:
            linear_out = self.base_linear(x)
            # 可学习的加权组合
            alpha = torch.sigmoid(self.combine_weight)
            return alpha * wavelet_out + (1 - alpha) * linear_out
        
        return wavelet_out


class WavKANProjectionHead(nn.Module):
    """
    基于 wav-KAN 的投影头。
    
    使用可学习的小波激活函数替代传统 MLP 中的固定激活函数，
    能够更好地逼近心音特征的复杂拓扑结构。
    
    结构：Input -> KANLinear(hidden) -> KANLinear(output)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        output_dim: int = 128,
        wavelet_type: Literal["mexican_hat", "morlet", "dog", "shannon"] = "morlet",
        num_layers: int = 2,
        num_wavelets: Optional[int] = None,
        use_base_linear: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出嵌入维度
            wavelet_type: 小波类型
            num_layers: 层数
            use_base_linear: 是否在 KAN 层中包含基础线性变换
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.wavelet_type = wavelet_type
        # Backward-compat: older versions exposed num_wavelets; current wav-KAN uses
        # a fixed (learned) wavelet transform per layer.
        self.num_wavelets = num_wavelets
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            
            layers.append(KANLinear(
                in_features=in_dim,
                out_features=out_dim,
                wavelet_type=wavelet_type,
                use_base_linear=use_base_linear
            ))
            
            # 中间层添加激活和 dropout
            if i < num_layers - 1:
                layers.append(nn.SiLU())  # 使用 SiLU 作为额外的非线性
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
            in_dim = out_dim
        
        self.kan = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入特征 (batch_size, input_dim)
            
        Returns:
            嵌入向量 (batch_size, output_dim)
        """
        return self.kan(x)
    
    def get_output_dim(self) -> int:
        """获取输出维度。"""
        return self.output_dim
    
    def get_wavelet_type(self) -> str:
        """获取小波类型。"""
        return self.wavelet_type

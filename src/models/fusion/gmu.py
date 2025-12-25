"""
Gated Multimodal Unit (GMU) for adaptive fusion of time and frequency streams.

Implements:
1. Upsampling alignment: Align sequence lengths via transposed convolution
2. Gated fusion: Dynamic weighting between modalities using sigmoid gating
"""

import torch
import torch.nn as nn
from typing import Optional, Literal


class GatedMultimodalUnit(nn.Module):
    """
    门控多模态融合单元 (GMU)。
    
    对于时频双流特征，使用 Sigmoid 门控机制动态选择
    在不同时间步上信赖哪个模态的特征。
    
    公式:
        G = σ(W_g · [Z_time; Z_freq])
        Z_fused = G ⊙ Z_time + (1-G) ⊙ Z_freq
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dim: 特征维度（假设两个流维度相同）
            hidden_dim: 门控网络隐藏层维度，None 表示使用 feature_dim
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        hidden_dim = hidden_dim or feature_dim
        
        # 门控网络：输入为拼接的双流特征
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()
        )
        
        # Layer normalization for output
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(
        self,
        z_time: torch.Tensor,
        z_freq: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            z_time: 时域流特征 (batch, seq_len, feature_dim)
            z_freq: 频域流特征 (batch, seq_len, feature_dim)，需与 z_time 长度一致
            
        Returns:
            融合特征 (batch, seq_len, feature_dim)
        """
        # 拼接双流特征
        concat = torch.cat([z_time, z_freq], dim=-1)  # (B, L, 2*D)
        
        # 计算门控权重
        gate = self.gate(concat)  # (B, L, D)
        
        # 加权融合
        z_fused = gate * z_time + (1 - gate) * z_freq  # (B, L, D)
        
        # Layer normalization
        z_fused = self.norm(z_fused)
        
        return z_fused


class AdaptiveGatedFusion(nn.Module):
    """
    自适应门控融合模块，包含序列长度对齐。
    
    用于融合不同长度的时域流和频域流特征：
    - 时域流：625 tokens (原始波形下采样)
    - 频域流：125 tokens (声谱图 patch embedding)
    
    使用转置卷积将频域流上采样到时域流长度，然后进行门控融合。
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        time_seq_len: int = 625,
        freq_seq_len: int = 125,
        upsample_kernel: int = 5,
        fusion_mode: Literal["gmu", "concat", "add"] = "gmu",
        dropout: float = 0.1,
        d_model: Optional[int] = None,
        mode: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            feature_dim: 特征维度
            time_seq_len: 时域流序列长度
            freq_seq_len: 频域流序列长度
            upsample_kernel: 上采样卷积核大小
            fusion_mode: 融合模式 ("gmu", "concat", "add")
            dropout: Dropout 概率
        """
        # Backward-compat aliases
        if d_model is not None:
            feature_dim = d_model
        if mode is not None:
            fusion_mode = mode  # type: ignore[assignment]

        super().__init__()
        
        self.feature_dim = feature_dim
        self.time_seq_len = time_seq_len
        self.freq_seq_len = freq_seq_len
        self.fusion_mode = fusion_mode
        
        # 计算上采样因子
        self.upsample_factor = time_seq_len // freq_seq_len  # 625 // 125 = 5
        
        # 频域流上采样：使用转置卷积
        # 输入: (B, D, freq_seq_len) -> 输出: (B, D, time_seq_len)
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=upsample_kernel,
                stride=self.upsample_factor,
                padding=upsample_kernel // 2,
                output_padding=self.upsample_factor - 1
            ),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 融合模块
        if fusion_mode == "gmu":
            self.fusion = GatedMultimodalUnit(
                feature_dim=feature_dim,
                dropout=dropout
            )
            self.output_dim = feature_dim
        elif fusion_mode == "concat":
            self.fusion = nn.Identity()
            self.output_dim = feature_dim * 2
            # 可选：添加投影层降维
            self.project = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.LayerNorm(feature_dim)
            )
        elif fusion_mode == "add":
            self.fusion = nn.Identity()
            self.output_dim = feature_dim
            self.norm = nn.LayerNorm(feature_dim)
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
    
    def forward(
        self,
        z_time: torch.Tensor,
        z_freq: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            z_time: 时域流特征 (batch, time_seq_len, feature_dim) 或 (batch, feature_dim, time_seq_len)
            z_freq: 频域流特征 (batch, freq_seq_len, feature_dim) 或 (batch, feature_dim, freq_seq_len)
            
        Returns:
            融合特征 (batch, time_seq_len, feature_dim)
        """
        # 确保输入格式为 (B, L, D)
        if z_time.shape[1] == self.feature_dim:
            z_time = z_time.transpose(1, 2)  # (B, D, L) -> (B, L, D)
        if z_freq.shape[1] == self.feature_dim:
            z_freq = z_freq.transpose(1, 2)
        
        # 上采样频域流
        z_freq_up = z_freq.transpose(1, 2)  # (B, L, D) -> (B, D, L)
        z_freq_up = self.upsample(z_freq_up)  # (B, D, time_seq_len)
        z_freq_up = z_freq_up.transpose(1, 2)  # (B, time_seq_len, D)
        
        # 确保长度匹配
        if z_freq_up.shape[1] != z_time.shape[1]:
            # 微调长度
            target_len = z_time.shape[1]
            if z_freq_up.shape[1] > target_len:
                z_freq_up = z_freq_up[:, :target_len, :]
            else:
                pad_len = target_len - z_freq_up.shape[1]
                z_freq_up = nn.functional.pad(z_freq_up, (0, 0, 0, pad_len))
        
        # 融合
        if self.fusion_mode == "gmu":
            z_fused = self.fusion(z_time, z_freq_up)
        elif self.fusion_mode == "concat":
            z_fused = torch.cat([z_time, z_freq_up], dim=-1)
            z_fused = self.project(z_fused)
        elif self.fusion_mode == "add":
            z_fused = self.norm(z_time + z_freq_up)
        
        return z_fused
    
    def get_output_dim(self) -> int:
        """获取输出特征维度。"""
        if self.fusion_mode == "concat" and not hasattr(self, 'project'):
            return self.feature_dim * 2
        return self.feature_dim

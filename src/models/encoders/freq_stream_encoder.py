"""
Frequency-Stream Encoder for spectrogram processing.

Processes Log-Mel spectrograms using vertical strip patch embedding
and bidirectional Mamba for capturing spectral patterns in heart sounds.

Architecture:
1. Vertical Strip Patch Embedding: Preserves frequency axis integrity
2. Learnable Positional Encoding: Maintains temporal order
3. BiMamba Backbone: Captures long-range spectral-temporal patterns
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Literal

from .mamba_encoder import MambaEncoder, BiMambaBlock, MambaBlock


class VerticalStripPatchEmbed(nn.Module):
    """
    垂直条带 Patch Embedding。
    
    借鉴 AST 的设计，但为保持频率维度的完整性（频率轴上的位置具有明确的物理意义），
    采用垂直条带切分策略。
    
    卷积核一次性覆盖所有频率 bins，并在时间轴上滑动。
    """
    
    def __init__(
        self,
        n_mels: int = 64,
        n_frames: int = 500,
        patch_width: int = 4,
        embed_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Args:
            n_mels: 梅尔频率 bins 数量
            n_frames: 时间帧数
            patch_width: 每个 patch 的时间宽度
            embed_dim: 嵌入维度
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.patch_width = patch_width
        self.embed_dim = embed_dim
        self.n_patches = n_frames // patch_width  # 输出 token 数
        
        # 垂直条带卷积：覆盖所有频率，时间上滑动
        # 输入: (B, 1, n_mels, n_frames) -> 输出: (B, embed_dim, 1, n_patches)
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=(n_mels, patch_width),
            stride=(n_mels, patch_width),
            padding=0
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 梅尔声谱图 (batch, n_mels, n_frames) 或 (batch, 1, n_mels, n_frames)
            
        Returns:
            Patch embeddings (batch, n_patches, embed_dim)
        """
        # 确保输入为 4D: (B, 1, n_mels, n_frames)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # 应用卷积
        x = self.proj(x)  # (B, embed_dim, 1, n_patches)
        
        # 重塑为序列格式
        x = x.squeeze(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        
        # Normalize and dropout
        x = self.norm(x)
        x = self.dropout(x)
        
        return x
    
    def get_num_patches(self) -> int:
        """获取输出 patch 数量。"""
        return self.n_patches


class LearnablePositionalEncoding(nn.Module):
    """
    可学习的一维位置编码。
    
    与固定的 sinusoidal 编码不同，可学习编码能够适应
    心音信号中周期性结构的特点。
    """
    
    def __init__(
        self,
        max_len: int = 1000,
        embed_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Args:
            max_len: 最大序列长度
            embed_dim: 嵌入维度
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码。
        
        Args:
            x: 输入序列 (batch, seq_len, embed_dim)
            
        Returns:
            带位置编码的序列 (batch, seq_len, embed_dim)
        """
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]
        return self.dropout(x)


class FreqStreamEncoder(nn.Module):
    """
    频域流编码器。
    
    用于处理 Log-Mel 声谱图，捕捉能量纹理分布，
    这对识别心脏杂音（如收缩期喷射性杂音的菱形包络）至关重要。
    
    架构:
    1. Vertical Strip Patch Embedding
    2. Learnable Positional Encoding
    3. BiMamba Backbone (4 layers)
    4. Global Average Pooling (optional)
    """
    
    def __init__(
        self,
        n_mels: int = 64,
        n_frames: int = 500,
        patch_width: int = 4,
        embed_dim: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_gap: bool = True,
        bidirectional: bool = True,
        bi_fusion_mode: Literal["concat", "gate", "add"] = "concat"
    ):
        """
        Args:
            n_mels: 梅尔频率 bins 数量
            n_frames: 时间帧数 (5s * 2kHz / hop_length)
            patch_width: 每个 patch 的时间宽度
            embed_dim: 嵌入维度
            n_layers: Mamba 层数
            d_state: SSM 状态维度
            d_conv: 局部卷积宽度
            expand: 内部维度扩展因子
            dropout: Dropout 概率
            use_gap: 是否使用 Global Average Pooling
            bidirectional: 是否使用双向 Mamba
            bi_fusion_mode: BiMamba 融合模式
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_gap = use_gap
        self.n_patches = n_frames // patch_width
        
        # Patch Embedding
        self.patch_embed = VerticalStripPatchEmbed(
            n_mels=n_mels,
            n_frames=n_frames,
            patch_width=patch_width,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # Positional Encoding
        self.pos_encode = LearnablePositionalEncoding(
            max_len=self.n_patches + 10,  # 留一些余量
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # Mamba Backbone
        if bidirectional:
            self.layers = nn.ModuleList([
                BiMambaBlock(
                    d_model=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    fusion_mode=bi_fusion_mode
                )
                for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                MambaBlock(
                    d_model=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout
                )
                for _ in range(n_layers)
            ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Global Average Pooling
        if self.use_gap:
            self.gap = nn.AdaptiveAvgPool1d(1)
    
    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 梅尔声谱图 (batch, n_mels, n_frames) 或 (batch, 1, n_mels, n_frames)
            return_sequence: 是否返回完整序列（用于融合）
            
        Returns:
            If use_gap=True and not return_sequence: (batch, embed_dim)
            Else: (batch, n_patches, embed_dim)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add positional encoding
        x = self.pos_encode(x)  # (B, n_patches, embed_dim)
        
        # Apply Mamba layers
        for layer in self.layers:
            x = layer(x)
        
        # Final norm
        x = self.norm(x)  # (B, n_patches, embed_dim)
        
        # Return sequence for fusion or apply GAP
        if return_sequence:
            return x
        
        if self.use_gap:
            x = x.transpose(1, 2)  # (B, embed_dim, n_patches)
            x = self.gap(x)  # (B, embed_dim, 1)
            x = x.squeeze(-1)  # (B, embed_dim)
        
        return x
    
    def get_output_dim(self) -> int:
        """获取输出特征维度。"""
        return self.embed_dim
    
    def get_num_tokens(self) -> int:
        """获取输出 token 数量。"""
        return self.n_patches


class FreqStreamEncoderWithProjection(nn.Module):
    """
    带投影头的频域流编码器。
    
    用于独立的频域流预训练或消融实验。
    """
    
    def __init__(
        self,
        n_mels: int = 64,
        n_frames: int = 500,
        patch_width: int = 4,
        embed_dim: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        bi_fusion_mode: str = "concat",
        projection_dim: int = 128,
        projection_hidden_dim: int = 512,
        **kwargs  # 忽略其他参数（如 _metadata_）
    ):
        super().__init__()
        
        self.encoder = FreqStreamEncoder(
            n_mels=n_mels,
            n_frames=n_frames,
            patch_width=patch_width,
            embed_dim=embed_dim,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            use_gap=True,
            bidirectional=bidirectional,
            bi_fusion_mode=bi_fusion_mode
        )
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_dim, projection_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False
    ) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 梅尔声谱图
            return_embedding: 是否返回投影前的嵌入
            
        Returns:
            投影后的特征或编码器嵌入
        """
        embedding = self.encoder(x)
        
        if return_embedding:
            return embedding
        
        return self.projection(embedding)
    
    def get_encoder(self) -> FreqStreamEncoder:
        """获取编码器（不含投影头）。"""
        return self.encoder
    
    def get_output_dim(self) -> int:
        """获取输出维度。"""
        return self.encoder.get_output_dim()

"""
Dual-Stream Encoder for time-frequency fusion.

Combines time-domain and frequency-domain streams with adaptive gated fusion.

Architecture:
1. Time-Stream: CNN Tokenizer + BiMamba (processes raw waveform)
2. Freq-Stream: Patch Embedding + BiMamba (processes mel-spectrogram)
3. Adaptive Gated Fusion: GMU for dynamic modality weighting
4. Optional Projection Head: MLP or wav-KAN
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Literal, Union

from .hybrid_encoder import HybridEncoder, ProjectionHead
from .freq_stream_encoder import FreqStreamEncoder
from ..fusion.gmu import AdaptiveGatedFusion
from ..heads.mlp_head import MLPProjectionHead


class DualStreamEncoder(nn.Module):
    """
    双流时频编码器。
    
    融合时域流和频域流的特征，用于心音信号的自监督表征学习。
    
    时域流：
    - 输入：原始波形 (B, 1, n_samples)
    - 架构：CNN Tokenizer + BiMamba
    - 输出：625 tokens @ 128-dim
    
    频域流：
    - 输入：梅尔声谱图 (B, n_mels, n_frames)
    - 架构：Vertical Patch Embed + BiMamba
    - 输出：125 tokens @ 128-dim
    
    融合：
    - 频域流上采样对齐至时域流长度
    - GMU 自适应门控融合
    """
    
    def __init__(
        self,
        time_stream_config: Dict[str, Any],
        freq_stream_config: Dict[str, Any],
        fusion_config: Optional[Dict[str, Any]] = None,
        projection_config: Optional[Dict[str, Any]] = None,
        use_gap: bool = True
    ):
        """
        Args:
            time_stream_config: 时域流配置 (CNN + Mamba)
            freq_stream_config: 频域流配置
            fusion_config: 融合模块配置
            projection_config: 投影头配置
            use_gap: 是否使用 Global Average Pooling
        """
        super().__init__()
        
        self.use_gap = use_gap
        
        # 时域流编码器 (复用 HybridEncoder)
        self.time_stream = HybridEncoder(
            cnn_config=time_stream_config.get('cnn_config', {}),
            mamba_config=time_stream_config.get('mamba_config', {}),
            projection_config=None,  # 不在单流加投影头
            use_gap=False  # 保留序列用于融合
        )
        
        # 频域流编码器
        self.freq_stream = FreqStreamEncoder(
            n_mels=freq_stream_config.get('n_mels', 64),
            n_frames=freq_stream_config.get('n_frames', 500),
            patch_width=freq_stream_config.get('patch_width', 4),
            embed_dim=freq_stream_config.get('embed_dim', 128),
            n_layers=freq_stream_config.get('n_layers', 4),
            d_state=freq_stream_config.get('d_state', 16),
            d_conv=freq_stream_config.get('d_conv', 4),
            expand=freq_stream_config.get('expand', 2),
            dropout=freq_stream_config.get('dropout', 0.1),
            use_gap=False,  # 保留序列用于融合
            bidirectional=freq_stream_config.get('bidirectional', True),
            bi_fusion_mode=freq_stream_config.get('bi_fusion_mode', 'concat')
        )
        
        # 获取特征维度和序列长度
        self.feature_dim = self.time_stream.get_output_dim()
        self.time_seq_len = self._compute_time_seq_len(time_stream_config)
        self.freq_seq_len = self.freq_stream.get_num_tokens()
        
        # 融合模块
        fusion_config = fusion_config or {}
        fusion_mode = fusion_config.get('mode', 'gmu')
        
        if fusion_mode == 'none':
            # 不融合，只使用时域流
            self.fusion = None
        else:
            self.fusion = AdaptiveGatedFusion(
                feature_dim=self.feature_dim,
                time_seq_len=self.time_seq_len,
                freq_seq_len=self.freq_seq_len,
                upsample_kernel=fusion_config.get('upsample_kernel', 5),
                fusion_mode=fusion_mode,
                dropout=fusion_config.get('dropout', 0.1)
            )
        
        # Global Average Pooling
        if use_gap:
            self.gap = nn.AdaptiveAvgPool1d(1)
        
        # 投影头 (可选)
        if projection_config is not None:
            proj_type = projection_config.get('type', 'mlp')
            if proj_type == 'mlp':
                self.projection = MLPProjectionHead(
                    input_dim=self.feature_dim,
                    hidden_dim=projection_config.get('hidden_dim', 512),
                    output_dim=projection_config.get('output_dim', 128),
                    num_layers=projection_config.get('num_layers', 2)
                )
            elif proj_type == 'wavkan':
                from ..heads.wavkan_head import WavKANProjectionHead
                self.projection = WavKANProjectionHead(
                    input_dim=self.feature_dim,
                    hidden_dim=projection_config.get('hidden_dim', 1024),
                    output_dim=projection_config.get('output_dim', 128),
                    wavelet_type=projection_config.get('wavelet_type', 'morlet'),
                    num_layers=projection_config.get('num_layers', 2)
                )
            else:
                raise ValueError(f"Unknown projection type: {proj_type}")
        else:
            self.projection = None
    
    def _compute_time_seq_len(self, config: Dict[str, Any]) -> int:
        """计算时域流的输出序列长度。"""
        # 默认配置: 10000 samples, strides=[4,4,1] -> 10000/16 = 625
        n_samples = config.get('n_samples', 10000)
        strides = config.get('cnn_config', {}).get('strides', [4, 4, 1])
        
        seq_len = n_samples
        for s in strides:
            seq_len = seq_len // s
        
        return seq_len
    
    def forward(
        self,
        waveform: torch.Tensor,
        mel_spec: Optional[torch.Tensor] = None,
        return_embedding: bool = False,
        return_streams: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        前向传播。
        
        Args:
            waveform: 原始波形 (batch, n_samples) 或 (batch, 1, n_samples)
            mel_spec: 梅尔声谱图 (batch, n_mels, n_frames)，可选
            return_embedding: 是否返回融合后的嵌入（投影前）
            return_streams: 是否返回各流的独立输出
            
        Returns:
            如果 return_streams=True:
                (z_fused, z_time, z_freq)
            如果有投影头且 return_embedding=False:
                投影后的嵌入 (batch, output_dim)
            否则:
                融合特征 (batch, feature_dim)
        """
        # 确保波形格式为 (B, 1, n_samples)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # (B, n_samples) -> (B, 1, n_samples)
        
        # 时域流
        z_time = self.time_stream(waveform, return_embedding=True)  # (B, seq, D) or (B, D)
        
        # 确保时域输出是序列格式
        if z_time.dim() == 2 and self.fusion is not None:
            # 如果已经池化，无法融合
            raise ValueError("Time stream must return sequence for fusion")
        
        # 频域流
        if mel_spec is not None and self.fusion is not None:
            z_freq = self.freq_stream(mel_spec, return_sequence=True)  # (B, seq, D)
            
            # 融合
            z_fused = self.fusion(z_time, z_freq)  # (B, time_seq, D)
        else:
            # 仅时域流
            z_fused = z_time
            z_freq = None
        
        # Global Average Pooling
        if self.use_gap:
            if z_fused.dim() == 3:
                z_fused = z_fused.transpose(1, 2)  # (B, D, seq)
                z_fused = self.gap(z_fused).squeeze(-1)  # (B, D)
        
        # 返回各流输出
        if return_streams:
            z_time_pooled = z_time
            if z_time.dim() == 3:
                z_time_pooled = z_time.mean(dim=1)
            z_freq_pooled = None
            if z_freq is not None:
                z_freq_pooled = z_freq.mean(dim=1) if z_freq.dim() == 3 else z_freq
            return z_fused, z_time_pooled, z_freq_pooled
        
        # 返回嵌入或投影
        if return_embedding or self.projection is None:
            return z_fused
        
        return self.projection(z_fused)
    
    def get_encoder(self) -> nn.Module:
        """获取编码器部分（不含投影头）。"""
        return nn.ModuleDict({
            'time_stream': self.time_stream,
            'freq_stream': self.freq_stream,
            'fusion': self.fusion
        })
    
    def get_output_dim(self) -> int:
        """获取输出特征维度。"""
        return self.feature_dim
    
    def get_time_stream(self) -> HybridEncoder:
        """获取时域流编码器。"""
        return self.time_stream
    
    def get_freq_stream(self) -> FreqStreamEncoder:
        """获取频域流编码器。"""
        return self.freq_stream


def build_dual_stream_encoder(
    time_stream: Optional[Dict[str, Any]] = None,
    freq_stream: Optional[Dict[str, Any]] = None,
    fusion: Optional[Dict[str, Any]] = None,
    projection: Optional[Dict[str, Any]] = None,
    use_gap: bool = True,
    **kwargs
) -> DualStreamEncoder:
    """
    从配置构建双流编码器（支持 Hydra instantiate）。
    
    Args:
        time_stream: 时域流配置
        freq_stream: 频域流配置
        fusion: 融合模块配置
        projection: 投影头配置
        use_gap: 是否使用 Global Average Pooling
        **kwargs: 其他参数（会被忽略）
        
    Returns:
        DualStreamEncoder 实例
    """
    return DualStreamEncoder(
        time_stream_config=time_stream or {},
        freq_stream_config=freq_stream or {},
        fusion_config=fusion or {},
        projection_config=projection,
        use_gap=use_gap
    )

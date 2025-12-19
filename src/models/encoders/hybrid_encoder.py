"""
Hybrid CNN-Mamba Encoder combining local and global modeling.

Architecture:
1. CNN Frontend: Extracts local morphological features (S1/S2 waveforms, murmurs)
2. Mamba Backend: Captures long-range rhythmic patterns and temporal evolution
3. Projection Head: Maps features to embedding space for contrastive learning
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .cnn_encoder import CNNEncoder
from .mamba_encoder import MambaEncoder


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.
    
    Maps encoder outputs to a lower-dimensional embedding space
    where contrastive loss is computed.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 2
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of layers (must be >= 2)
        """
        super().__init__()
        
        assert num_layers >= 2, "ProjectionHead must have at least 2 layers"
        
        layers = []
        
        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ])
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ])
        
        # Output layer (no activation)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features of shape (batch_size, input_dim)
            
        Returns:
            Embeddings of shape (batch_size, output_dim)
        """
        return self.mlp(x)


class HybridEncoder(nn.Module):
    """
    Hybrid CNN-Mamba encoder for heart sound representation learning.
    
    Combines:
    - CNN for local feature extraction (short-term patterns)
    - Mamba for global temporal modeling (long-term dependencies)
    - Optional projection head for contrastive learning
    
    Supports both unidirectional and bidirectional Mamba.
    """
    
    def __init__(
        self,
        cnn_config: Dict[str, Any],
        mamba_config: Dict[str, Any],
        projection_config: Optional[Dict[str, Any]] = None,
        use_gap: bool = True,
        **kwargs  # 忽略其他参数（如 _metadata_）
    ):
        """
        Args:
            cnn_config: Configuration for CNN encoder
            mamba_config: Configuration for Mamba encoder
                - bidirectional: bool, whether to use BiMamba
                - bi_fusion_mode: str, fusion mode for BiMamba ("concat", "gate", "add")
            projection_config: Optional configuration for projection head
            use_gap: Whether to use Global Average Pooling
        """
        super().__init__()
        
        self.use_gap = use_gap
        
        # CNN Frontend
        self.cnn = CNNEncoder(
            in_channels=cnn_config.get('in_channels', 1),
            out_channels=cnn_config.get('out_channels', [32, 64, 128]),
            kernel_sizes=cnn_config.get('kernel_sizes', [16, 8, 4]),
            strides=cnn_config.get('strides', [2, 2, 2]),
            use_batchnorm=cnn_config.get('use_batchnorm', True),
            dropout=cnn_config.get('dropout', 0.1),
            use_gap=False  # Don't pool in CNN, Mamba will handle sequence
        )
        
        # Mamba Backend (支持双向)
        cnn_out_dim = self.cnn.get_output_dim()
        self.mamba = MambaEncoder(
            d_model=mamba_config.get('d_model', 128),
            d_state=mamba_config.get('d_state', 16),
            d_conv=mamba_config.get('d_conv', 4),
            expand=mamba_config.get('expand', 2),
            n_layers=mamba_config.get('n_layers', 4),
            dropout=mamba_config.get('dropout', 0.1),
            use_gap=use_gap,
            input_projection=True,
            input_dim=cnn_out_dim,
            bidirectional=mamba_config.get('bidirectional', False),
            bi_fusion_mode=mamba_config.get('bi_fusion_mode', 'concat')
        )
        
        # Projection Head (optional, for MoCo)
        if projection_config is not None:
            mamba_out_dim = self.mamba.get_output_dim()
            self.projection = ProjectionHead(
                input_dim=projection_config.get('input_dim', mamba_out_dim),
                hidden_dim=projection_config.get('hidden_dim', 128),
                output_dim=projection_config.get('output_dim', 128),
                num_layers=projection_config.get('num_layers', 2)
            )
        else:
            self.projection = None
    
    def forward(self, x, return_embedding=False):
        """
        Forward pass through hybrid encoder.
        
        Args:
            x: Input audio of shape (batch_size, 1, n_samples)
            return_embedding: If True, return encoder output before projection
            
        Returns:
            If projection head exists and return_embedding=False:
                Projected embeddings (batch_size, output_dim)
            If return_embedding=True:
                Encoder features (batch_size, d_model)
        """
        # CNN: Extract local features
        cnn_features = self.cnn(x)  # (batch, channels, seq_len')
        
        # Mamba: Model temporal dependencies
        mamba_features = self.mamba(cnn_features)  # (batch, d_model) or (batch, seq_len, d_model)
        
        # Return encoder output if requested
        if return_embedding or self.projection is None:
            return mamba_features
        
        # Apply projection head
        embeddings = self.projection(mamba_features)  # (batch, output_dim)
        
        return embeddings
    
    def get_encoder(self):
        """
        Get the encoder (without projection head) for downstream tasks.
        
        Returns:
            nn.Module wrapping CNN + Mamba
        """
        return nn.Sequential(self.cnn, self.mamba)
    
    def get_output_dim(self):
        """Get output dimension of the encoder."""
        return self.mamba.get_output_dim()


def build_hybrid_encoder(config: Dict[str, Any]) -> HybridEncoder:
    """
    Build HybridEncoder from configuration dictionary.
    
    Args:
        config: Configuration dictionary with cnn_config, mamba_config, etc.
        
    Returns:
        HybridEncoder instance
    """
    return HybridEncoder(
        cnn_config=config.get('cnn_config', {}),
        mamba_config=config.get('mamba_config', {}),
        projection_config=config.get('projection_config'),
        use_gap=config.get('use_gap', True)
    )

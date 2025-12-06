"""
1D CNN Encoder for heart sound signal processing.

Extracts local morphological features from PCG signals.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class Conv1DBlock(nn.Module):
    """
    Basic 1D convolutional block with Conv -> BatchNorm -> ReLU -> Dropout.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        use_batchnorm: bool = True,
        dropout: float = 0.0
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size
            stride: Stride
            padding: Padding (auto-calculated if None)
            use_batchnorm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        ]
        
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class CNNEncoder(nn.Module):
    """
    1D CNN encoder for extracting local features from PCG signals.
    
    Architecture:
    - Multiple convolutional blocks with progressive channel expansion
    - Downsampling via strided convolutions
    - Global Average Pooling to handle variable-length sequences
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [16, 8, 4],
        strides: List[int] = [2, 2, 2],
        use_batchnorm: bool = True,
        dropout: float = 0.1,
        use_gap: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels (1 for mono audio)
            out_channels: List of output channels for each conv block
            kernel_sizes: List of kernel sizes for each conv block
            strides: List of strides for each conv block
            use_batchnorm: Whether to use batch normalization
            dropout: Dropout probability
            use_gap: Whether to use Global Average Pooling
        """
        super().__init__()
        
        assert len(out_channels) == len(kernel_sizes) == len(strides), \
            "Length mismatch between out_channels, kernel_sizes, and strides"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_gap = use_gap
        self.final_channels = out_channels[-1]
        
        # Build convolutional blocks
        conv_blocks = []
        in_ch = in_channels
        
        for out_ch, kernel_size, stride in zip(out_channels, kernel_sizes, strides):
            conv_blocks.append(Conv1DBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                use_batchnorm=use_batchnorm,
                dropout=dropout
            ))
            in_ch = out_ch
        
        self.conv_blocks = nn.Sequential(*conv_blocks)
        
        # Global Average Pooling
        if self.use_gap:
            self.gap = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_length)
            
        Returns:
            If use_gap=True: (batch_size, final_channels)
            If use_gap=False: (batch_size, final_channels, seq_length')
        """
        # Apply convolutional blocks
        x = self.conv_blocks(x)  # (batch, channels, seq_length')
        
        # Apply GAP if enabled
        if self.use_gap:
            x = self.gap(x)  # (batch, channels, 1)
            x = x.squeeze(-1)  # (batch, channels)
        
        return x
    
    def get_output_dim(self):
        """Get output dimension."""
        return self.final_channels

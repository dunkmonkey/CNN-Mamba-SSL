"""
Mamba State Space Model Encoder for long-range temporal modeling.

Processes sequential features with linear complexity using selective state spaces.
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available. Install with: pip install mamba-ssm")


class MambaBlock(nn.Module):
    """
    Single Mamba block with residual connection and layer norm.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0
    ):
        """
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Local convolution width
            expand: Expansion factor
            dropout: Dropout probability
        """
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm is not installed. Install with: pip install mamba-ssm causal-conv1d"
            )
        
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        """
        Forward pass with residual connection.
        
        Args:
            x: Input of shape (batch, seq_len, d_model)
            
        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        x = x + residual
        return x


class MambaEncoder(nn.Module):
    """
    Mamba encoder for processing sequential features.
    
    Captures long-range dependencies with linear computational complexity.
    Suitable for analyzing heart rhythm patterns and temporal evolution.
    """
    
    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        dropout: float = 0.1,
        use_gap: bool = True,
        input_projection: bool = False,
        input_dim: Optional[int] = None
    ):
        """
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Local convolution width
            expand: Expansion factor
            n_layers: Number of Mamba blocks
            dropout: Dropout probability
            use_gap: Whether to use Global Average Pooling
            input_projection: Whether to project input to d_model
            input_dim: Input dimension (required if input_projection=True)
        """
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm is not installed. Install with: pip install mamba-ssm causal-conv1d"
            )
        
        self.d_model = d_model
        self.use_gap = use_gap
        
        # Input projection if needed
        if input_projection:
            assert input_dim is not None, "input_dim must be provided when input_projection=True"
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Identity()
        
        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Global Average Pooling
        if self.use_gap:
            self.gap = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch, channels, seq_len) or (batch, seq_len, d_model)
            
        Returns:
            If use_gap=True: (batch, d_model)
            If use_gap=False: (batch, seq_len, d_model)
        """
        # Handle input format
        if x.dim() == 3 and x.shape[1] < x.shape[2]:
            # Assume input is (batch, channels, seq_len), convert to (batch, seq_len, channels)
            x = x.transpose(1, 2)
        
        # Input projection if needed
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Apply Mamba blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final norm
        x = self.norm(x)  # (batch, seq_len, d_model)
        
        # Apply GAP if enabled
        if self.use_gap:
            x = x.transpose(1, 2)  # (batch, d_model, seq_len)
            x = self.gap(x)  # (batch, d_model, 1)
            x = x.squeeze(-1)  # (batch, d_model)
        
        return x
    
    def get_output_dim(self):
        """Get output dimension."""
        return self.d_model

"""Encoder architectures (CNN, Mamba, Hybrid, Frequency-Stream, Dual-Stream)."""

from .cnn_encoder import CNNEncoder, Conv1DBlock
from .mamba_encoder import MambaEncoder, MambaBlock, BiMambaBlock
from .hybrid_encoder import HybridEncoder, ProjectionHead
from .freq_stream_encoder import (
    FreqStreamEncoder,
    FreqStreamEncoderWithProjection,
    VerticalStripPatchEmbed,
    LearnablePositionalEncoding
)
from .dual_stream_encoder import DualStreamEncoder, build_dual_stream_encoder

__all__ = [
    # CNN
    "CNNEncoder",
    "Conv1DBlock",
    # Mamba
    "MambaEncoder",
    "MambaBlock",
    "BiMambaBlock",
    # Hybrid (Time-stream)
    "HybridEncoder",
    "ProjectionHead",
    # Frequency-stream
    "FreqStreamEncoder",
    "FreqStreamEncoderWithProjection",
    "VerticalStripPatchEmbed",
    "LearnablePositionalEncoding",
    # Dual-stream
    "DualStreamEncoder",
    "build_dual_stream_encoder",
]

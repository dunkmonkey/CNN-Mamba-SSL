"""Model architectures and modules."""

from .encoders import (
    CNNEncoder,
    MambaEncoder,
    MambaBlock,
    BiMambaBlock,
    HybridEncoder,
    FreqStreamEncoder,
    FreqStreamEncoderWithProjection,
    DualStreamEncoder,
    build_dual_stream_encoder,
)
from .moco_module import MoCoModule
from .dual_modal_moco_module import DualModalMoCoModule
from .classifier_module import ClassifierModule

# Fusion modules
from .fusion import GatedMultimodalUnit, AdaptiveGatedFusion

# Projection heads
from .heads import MLPProjectionHead, WavKANProjectionHead

__all__ = [
    # Encoders
    "CNNEncoder",
    "MambaEncoder",
    "MambaBlock",
    "BiMambaBlock",
    "HybridEncoder",
    "FreqStreamEncoder",
    "FreqStreamEncoderWithProjection",
    "DualStreamEncoder",
    "build_dual_stream_encoder",
    # Training modules
    "MoCoModule",
    "DualModalMoCoModule",
    "ClassifierModule",
    # Fusion
    "GatedMultimodalUnit",
    "AdaptiveGatedFusion",
    # Heads
    "MLPProjectionHead",
    "WavKANProjectionHead",
]

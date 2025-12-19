"""
Projection heads for contrastive learning.
"""

from .mlp_head import MLPProjectionHead
from .wavkan_head import WavKANProjectionHead

__all__ = ["MLPProjectionHead", "WavKANProjectionHead"]

#!/usr/bin/env python
"""
Export pretrained encoder from MoCo checkpoint.

Usage:
    python scripts/export_encoder.py checkpoint=path/to/moco_checkpoint.ckpt output=path/to/encoder.pth
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch

from src.models.moco_module import MoCoModule


def export_encoder(checkpoint_path: str, output_path: str):
    """
    Export encoder from MoCo checkpoint.
    
    Args:
        checkpoint_path: Path to MoCo checkpoint
        output_path: Path to save encoder weights
    """
    print("="*60)
    print("Exporting Encoder from MoCo Checkpoint")
    print("="*60)
    print(f"Input checkpoint: {checkpoint_path}")
    print(f"Output path: {output_path}")
    
    # Load MoCo module
    print("\nLoading MoCo checkpoint...")
    moco_module = MoCoModule.load_from_checkpoint(checkpoint_path)
    
    # Extract encoder (without projection head)
    print("Extracting encoder...")
    encoder = moco_module.get_encoder()
    
    # Save encoder weights
    print("Saving encoder weights...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(encoder.state_dict(), output_path)
    
    print(f"\nEncoder successfully exported to: {output_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Export encoder from MoCo checkpoint")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to MoCo checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output path for encoder weights')
    
    args = parser.parse_args()
    
    export_encoder(args.checkpoint, args.output)


if __name__ == "__main__":
    main()

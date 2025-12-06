#!/usr/bin/env python
"""
Evaluation script for trained models.

Usage:
    python scripts/evaluate.py checkpoint=path/to/checkpoint.ckpt
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.classifier_module import ClassifierModule
from src.data.datamodules.finetune_datamodule import FinetuneDataModule


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to: {save_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main evaluation function.
    
    Args:
        cfg: Hydra configuration
    """
    # Check for checkpoint path
    if 'checkpoint' not in cfg:
        raise ValueError("Please provide checkpoint path: python evaluate.py checkpoint=path/to/ckpt")
    
    checkpoint_path = cfg.checkpoint
    
    print("="*60)
    print("Evaluation Configuration:")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print("="*60)
    
    # Set seed
    L.seed_everything(cfg.seed, workers=True)
    
    # Load model from checkpoint
    print("\nLoading model from checkpoint...")
    model = ClassifierModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Build data module
    print("\nBuilding data module...")
    # Override with finetune data config if needed
    if 'data' not in cfg or 'finetune' not in str(cfg.data):
        cfg.data = OmegaConf.load(Path(__file__).parent.parent / "configs/data/finetune.yaml")
    
    datamodule = FinetuneDataModule(
        data_root=cfg.data.data_root,
        target_sr=cfg.data.target_sr,
        segment_length=cfg.data.segment_length,
        n_samples=cfg.data.n_samples,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        num_classes=cfg.data.num_classes,
        use_augmentation=False,  # No augmentation for evaluation
        use_cache=cfg.data.use_cache,
        cache_dir=cfg.data.get('cache_dir'),
        seed=cfg.seed
    )
    
    datamodule.setup('test')
    
    # Trainer for evaluation
    trainer = L.Trainer(
        accelerator='auto',
        devices=1,
        logger=False
    )
    
    # Test
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    test_results = trainer.test(model, datamodule=datamodule)
    
    # Collect predictions for detailed analysis
    print("\nCollecting predictions for detailed analysis...")
    all_preds = []
    all_targets = []
    all_probs = []
    
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            x, y = batch
            x = x.to(model.device)
            
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Classification report
    print("\n" + "="*60)
    print("Detailed Classification Report:")
    print("="*60)
    class_names = ['Normal', 'Abnormal'] if cfg.data.num_classes == 2 else [f'Class {i}' for i in range(cfg.data.num_classes)]
    print(classification_report(all_targets, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save confusion matrix plot
    output_dir = Path(cfg.get('output_dir', 'outputs/evaluation'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_confusion_matrix(cm, class_names, output_dir / 'confusion_matrix.png')
    
    # Save predictions
    results_file = output_dir / 'predictions.npz'
    np.savez(
        results_file,
        predictions=all_preds,
        targets=all_targets,
        probabilities=all_probs
    )
    print(f"\nPredictions saved to: {results_file}")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()

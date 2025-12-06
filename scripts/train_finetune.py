#!/usr/bin/env python
"""
Training script for fine-tuning on labeled heart sound data.

Usage:
    python scripts/train_finetune.py pretrained_encoder=path/to/encoder.pth
    python scripts/train_finetune.py trainer.freeze_encoder=true  # Linear probing
    python scripts/train_finetune.py trainer.freeze_encoder=false # Full fine-tuning
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from src.models.encoders.hybrid_encoder import build_hybrid_encoder
from src.models.classifier_module import ClassifierModule
from src.data.datamodules.finetune_datamodule import FinetuneDataModule


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main fine-tuning function.
    
    Args:
        cfg: Hydra configuration
    """
    # Override with finetune-specific defaults
    if 'trainer' not in cfg or 'finetune' not in str(cfg.trainer):
        cfg.trainer = OmegaConf.load(Path(__file__).parent.parent / "configs/trainer/finetune.yaml")
    if 'data' not in cfg or 'finetune' not in str(cfg.data):
        cfg.data = OmegaConf.load(Path(__file__).parent.parent / "configs/data/finetune.yaml")
    
    # Print configuration
    print("="*60)
    print("Fine-tuning Configuration:")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))
    print("="*60)
    
    # Set seed
    L.seed_everything(cfg.seed, workers=True)
    
    # Build encoder
    print("\nBuilding encoder...")
    encoder = build_hybrid_encoder(dict(cfg.model))
    
    # Load pretrained weights if provided
    if 'pretrained_encoder' in cfg and cfg.pretrained_encoder:
        print(f"\nLoading pretrained encoder from: {cfg.pretrained_encoder}")
        encoder.load_state_dict(torch.load(cfg.pretrained_encoder))
        print("Pretrained weights loaded successfully!")
    else:
        print("\nWarning: No pretrained encoder provided. Training from scratch.")
    
    # Get encoder without projection head for downstream task
    encoder_backbone = encoder.get_encoder()
    
    # Build classifier module
    print("\nBuilding classifier module...")
    classifier_module = ClassifierModule(
        encoder=encoder_backbone,
        num_classes=cfg.data.num_classes,
        hidden_dim=None,  # Simple linear classifier
        freeze_encoder=cfg.trainer.freeze_encoder,
        encoder_lr_multiplier=cfg.trainer.encoder_lr_multiplier,
        learning_rate=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        scheduler_config=dict(cfg.optimizer.scheduler) if hasattr(cfg.optimizer, 'scheduler') else None
    )
    
    print(f"Fine-tuning strategy: {'Linear Probing' if cfg.trainer.freeze_encoder else 'Full Fine-tuning'}")
    
    # Build data module
    print("\nBuilding data module...")
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
        use_augmentation=cfg.data.get('use_augmentation', True),
        augmentation_config=cfg.augmentation if hasattr(cfg, 'augmentation') else None,
        use_cache=cfg.data.use_cache,
        cache_dir=cfg.data.get('cache_dir'),
        seed=cfg.seed
    )
    
    # Setup callbacks
    print("\nSetting up callbacks...")
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.trainer.callbacks.model_checkpoint.monitor,
        mode=cfg.trainer.callbacks.model_checkpoint.mode,
        save_top_k=cfg.trainer.callbacks.model_checkpoint.save_top_k,
        save_last=cfg.trainer.callbacks.model_checkpoint.save_last,
        filename=cfg.trainer.callbacks.model_checkpoint.filename,
        auto_insert_metric_name=cfg.trainer.callbacks.model_checkpoint.auto_insert_metric_name,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor=cfg.trainer.callbacks.early_stopping.monitor,
        mode=cfg.trainer.callbacks.early_stopping.mode,
        patience=cfg.trainer.callbacks.early_stopping.patience,
        min_delta=cfg.trainer.callbacks.early_stopping.min_delta,
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(
        logging_interval=cfg.trainer.callbacks.lr_monitor.logging_interval
    )
    callbacks.append(lr_monitor)
    
    # Logger
    print("\nSetting up logger...")
    logger = TensorBoardLogger(
        save_dir=cfg.trainer.logger.tensorboard.save_dir,
        name=cfg.trainer.logger.tensorboard.name,
        version=cfg.trainer.logger.tensorboard.version,
        default_hp_metric=cfg.trainer.logger.tensorboard.default_hp_metric
    )
    
    # Trainer
    print("\nSetting up trainer...")
    trainer = L.Trainer(
        accelerator=cfg.trainer.trainer.accelerator,
        devices=cfg.trainer.trainer.devices,
        max_epochs=cfg.trainer.trainer.max_epochs,
        precision=cfg.trainer.trainer.precision,
        gradient_clip_val=cfg.trainer.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.trainer.trainer.log_every_n_steps,
        enable_checkpointing=cfg.trainer.trainer.enable_checkpointing,
        default_root_dir=cfg.trainer.trainer.default_root_dir,
        benchmark=cfg.trainer.trainer.benchmark,
        deterministic=cfg.trainer.trainer.deterministic,
        callbacks=callbacks,
        logger=logger
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting fine-tuning...")
    print("="*60)
    trainer.fit(classifier_module, datamodule=datamodule)
    
    # Test
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    trainer.test(classifier_module, datamodule=datamodule, ckpt_path='best')
    
    # Final summary
    print("\n" + "="*60)
    print("Fine-tuning complete!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print("="*60)


if __name__ == "__main__":
    main()

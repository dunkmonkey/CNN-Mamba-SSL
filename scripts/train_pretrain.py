#!/usr/bin/env python
"""
Training script for MoCo pretraining on heart sound data.

Usage:
    python scripts/train_pretrain.py
    python scripts/train_pretrain.py experiment=exp001_baseline
    python scripts/train_pretrain.py model.mamba_config.n_layers=6
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
from src.models.moco_module import MoCoModule
from src.data.datamodules.pretrain_datamodule import PretrainDataModule
from src.callbacks.momentum_update import MomentumUpdateCallback


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main training function.
    
    Args:
        cfg: Hydra configuration
    """
    # Print configuration
    print("="*60)
    print("Configuration:")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))
    print("="*60)
    
    # Set seed for reproducibility
    L.seed_everything(cfg.seed, workers=True)
    
    # Build encoder
    print("\nBuilding encoder...")
    encoder = build_hybrid_encoder(dict(cfg.model))
    print(f"Encoder output dimension: {encoder.get_output_dim()}")
    
    # Build MoCo module
    print("\nBuilding MoCo module...")
    moco_module = MoCoModule(
        encoder=encoder,
        encoder_config=cfg.model,
        queue_size=cfg.get('moco', {}).get('queue_size', 65536),
        momentum=cfg.get('moco', {}).get('momentum', 0.999),
        temperature=cfg.get('moco', {}).get('temperature', 0.07),
        learning_rate=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        scheduler_config=dict(cfg.optimizer.scheduler) if hasattr(cfg.optimizer, 'scheduler') else None
    )
    
    # Build data module
    print("\nBuilding data module...")
    datamodule = PretrainDataModule(
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
        persistent_workers=cfg.data.get('persistent_workers', True),
        use_dual_augmentation=cfg.data.use_dual_augmentation,
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
    
    # Momentum update callback
    momentum_callback = MomentumUpdateCallback(
        momentum=cfg.get('moco', {}).get('momentum', 0.999)
    )
    callbacks.append(momentum_callback)
    
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
    print("Starting training...")
    print("="*60)
    trainer.fit(moco_module, datamodule=datamodule)
    
    # Save final encoder
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print("="*60)
    
    # Extract and save encoder
    encoder_save_path = Path(checkpoint_callback.dirpath) / "encoder_final.pth"
    torch.save(moco_module.encoder_q.state_dict(), encoder_save_path)
    print(f"Encoder weights saved to: {encoder_save_path}")


if __name__ == "__main__":
    main()

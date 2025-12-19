"""
Training script for dual-modal (time + frequency) pretraining with MoCo.

Example usage:
    # Phase 3: Dual-stream with GMU fusion
    python scripts/train_dual_pretrain.py experiment=ablation_c3 model=dual_stream
    
    # Phase 3: Dual-stream with GMU + wav-KAN
    python scripts/train_dual_pretrain.py experiment=ablation_c3 model=dual_stream_wavkan
    
    # Custom configuration
    python scripts/train_dual_pretrain.py model=dual_stream data=physionet2016_dual \
        trainer.max_epochs=300 data.batch_size=256
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from pathlib import Path
import torch

from src.data.datamodules.dual_modal_datamodule import DualModalPretrainDataModule
from src.models.encoders.dual_stream_encoder import build_dual_stream_encoder
from src.models.dual_modal_moco_module import DualModalMoCoModule
from src.callbacks.momentum_update import MomentumUpdateCallback


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main training function for dual-modal pretraining.
    """
    print("=" * 80)
    print("Dual-Modal MoCo Pretraining")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed for reproducibility
    if 'seed' in cfg:
        L.seed_everything(cfg.seed)
    
    # Initialize DataModule
    print("\n[1/4] Initializing DataModule...")
    datamodule = hydra.utils.instantiate(cfg.data)
    print(f"  - Dataset: {cfg.data.data_root}")
    print(f"  - Batch size: {cfg.data.batch_size}")
    print(f"  - Target SR: {cfg.data.target_sr} Hz")
    print(f"  - Mel bands: {cfg.data.n_mels}")
    print(f"  - Time augmentation: {cfg.data.apply_time_aug}")
    print(f"  - Freq augmentation: {cfg.data.apply_freq_aug}")
    
    # Build encoder
    print("\n[2/4] Building Encoder...")
    encoder = build_dual_stream_encoder(cfg.model)
    print(f"  - Model: {cfg.model._target_}")
    print(f"  - Output dim: {encoder.get_output_dim()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Initialize MoCo module
    print("\n[3/4] Initializing MoCo Module...")
    moco_config = cfg.get('moco', {})
    moco_module = DualModalMoCoModule(
        encoder=encoder,
        encoder_config=cfg.model,
        queue_size=moco_config.get('queue_size', 65536),
        momentum=moco_config.get('momentum', 0.999),
        temperature=moco_config.get('temperature', 0.07),
        learning_rate=cfg.optimizer.get('lr', 0.001),
        weight_decay=cfg.optimizer.get('weight_decay', 1e-4),
        scheduler_config=cfg.get('scheduler', None)
    )
    print(f"  - Queue size: {moco_config.get('queue_size', 65536)}")
    print(f"  - Momentum: {moco_config.get('momentum', 0.999)}")
    print(f"  - Temperature: {moco_config.get('temperature', 0.07)}")
    print(f"  - Learning rate: {cfg.optimizer.get('lr', 0.001)}")
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_dir = Path(cfg.get('output_dir', 'outputs')) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = L.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='p3_dual_{epoch:03d}_{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if cfg.trainer.get('early_stopping', False):
        early_stop_callback = L.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=cfg.trainer.get('patience', 20),
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    callbacks.append(L.callbacks.LearningRateMonitor(logging_interval='epoch'))
    
    # Momentum update callback (updates key encoder)
    callbacks.append(MomentumUpdateCallback())
    
    # Initialize Trainer
    print("\n[4/4] Initializing Trainer...")
    trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_config['callbacks'] = callbacks
    
    # Add logger if specified
    if 'logger' in cfg:
        logger = hydra.utils.instantiate(cfg.logger.wandb)
        trainer_config['logger'] = logger
        print(f"  - Logger: W&B (project={cfg.logger.wandb.project})")
    
    trainer = L.Trainer(**trainer_config)
    
    print(f"  - Max epochs: {cfg.trainer.max_epochs}")
    print(f"  - Precision: {cfg.trainer.get('precision', '32-true')}")
    print(f"  - Accelerator: {cfg.trainer.get('accelerator', 'auto')}")
    print(f"  - Devices: {cfg.trainer.get('devices', 'auto')}")
    
    # Train
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80 + "\n")
    
    trainer.fit(moco_module, datamodule=datamodule)
    
    # Save final encoder
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    
    encoder_path = Path(cfg.get('output_dir', 'outputs')) / 'encoder_final.pth'
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(moco_module.get_encoder().state_dict(), encoder_path)
    print(f"\nFinal encoder saved to: {encoder_path}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    
    return moco_module


if __name__ == "__main__":
    main()

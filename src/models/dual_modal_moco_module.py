"""
Dual-Modal MoCo Module for DualStreamEncoder.

Extends MoCo to handle both time-domain and frequency-domain inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Dict, Any, Optional, Tuple
from omegaconf import DictConfig
import copy

from ..losses.info_nce import MoCoLoss, compute_accuracy


class DualModalMoCoModule(L.LightningModule):
    """
    MoCo framework for dual-stream (time + frequency) self-supervised learning.
    
    Accepts dual-modal inputs:
    - Time-domain: (B, n_samples)
    - Frequency-domain: (B, n_mels, time_frames)
    
    Encoder must support dual-modal forward: encoder(time, freq)
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        encoder_config: DictConfig,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.07,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        scheduler_config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            encoder: Query encoder (DualStreamEncoder)
            encoder_config: Configuration for building momentum encoder
            queue_size: Size of negative sample queue
            momentum: Momentum coefficient for key encoder update
            temperature: Temperature for InfoNCE loss
            learning_rate: Learning rate
            weight_decay: Weight decay
            scheduler_config: Optional LR scheduler config
        """
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        
        # Query encoder (updated by gradients)
        self.encoder_q = encoder
        
        # Momentum encoder (updated by momentum)
        self.encoder_k = copy.deepcopy(encoder)
        
        # Freeze momentum encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        
        # MoCo hyperparameters
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config
        
        # Loss function
        self.criterion = MoCoLoss(temperature=temperature)
        
        # Initialize queue
        self.register_buffer("queue", torch.randn(queue_size, self.encoder_q.get_output_dim()))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Update momentum encoder: θ_k = m * θ_k + (1 - m) * θ_q
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """
        Update queue with new keys.
        
        Args:
            keys: (batch_size, dim)
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace oldest entries
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
        else:
            remaining = self.queue_size - ptr
            self.queue[ptr:] = keys[:remaining]
            self.queue[:batch_size - remaining] = keys[remaining:]
        
        # Update pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x_time: torch.Tensor, x_freq: torch.Tensor) -> Tuple:
        """
        Batch shuffle for DDP with dual-modal inputs.
        
        Returns:
            (x_time_shuffled, x_freq_shuffled, idx_unshuffle)
        """
        if not self.trainer or self.trainer.world_size == 1:
            return x_time, x_freq, None
        
        batch_size_this = x_time.shape[0]
        
        # Gather from all GPUs
        x_time_gather = self.all_gather(x_time, sync_grads=False)
        x_freq_gather = self.all_gather(x_freq, sync_grads=False)
        
        batch_size_all = x_time_gather.shape[0] * x_time_gather.shape[1]
        x_time_gather = x_time_gather.reshape(batch_size_all, *x_time.shape[1:])
        x_freq_gather = x_freq_gather.reshape(batch_size_all, *x_freq.shape[1:])
        
        # Random shuffle
        idx_shuffle = torch.randperm(batch_size_all, device=x_time.device)
        torch.distributed.broadcast(idx_shuffle, src=0)
        
        idx_unshuffle = torch.argsort(idx_shuffle)
        
        # Index for this GPU
        gpu_idx = self.trainer.global_rank
        idx_this = idx_shuffle.view(-1, batch_size_this)[gpu_idx]
        
        return x_time_gather[idx_this], x_freq_gather[idx_this], idx_unshuffle
    
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x: torch.Tensor, idx_unshuffle: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Undo batch shuffle.
        """
        if idx_unshuffle is None or not self.trainer or self.trainer.world_size == 1:
            return x
        
        batch_size_this = x.shape[0]
        
        x_gather = self.all_gather(x, sync_grads=False)
        batch_size_all = x_gather.shape[0] * x_gather.shape[1]
        x_gather = x_gather.reshape(batch_size_all, *x.shape[1:])
        
        gpu_idx = self.trainer.global_rank
        idx_this = idx_unshuffle.view(-1, batch_size_this)[gpu_idx]
        
        return x_gather[idx_this]
    
    def forward(
        self,
        time_q: torch.Tensor,
        freq_q: torch.Tensor,
        time_k: torch.Tensor,
        freq_k: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dual-modal inputs.
        
        Args:
            time_q: Query time-domain (B, n_samples)
            freq_q: Query frequency-domain (B, n_mels, frames)
            time_k: Key time-domain (B, n_samples)
            freq_k: Key frequency-domain (B, n_mels, frames)
        
        Returns:
            {'query': q, 'key': k, 'queue': queue}
        """
        # Compute query features
        q = self.encoder_q(time_q, freq_q, return_embedding=False)  # (B, dim)
        q = F.normalize(q, dim=1)
        
        # Compute key features (no gradient)
        with torch.no_grad():
            # Update momentum encoder
            self._momentum_update_key_encoder()
            
            # Shuffle for BN in DDP
            time_k, freq_k, idx_unshuffle = self._batch_shuffle_ddp(time_k, freq_k)
            
            # Compute keys
            k = self.encoder_k(time_k, freq_k, return_embedding=False)  # (B, dim)
            k = F.normalize(k, dim=1)
            
            # Unshuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        
        return {'query': q, 'key': k, 'queue': self.queue.clone().detach()}
    
    def training_step(self, batch, batch_idx):
        """
        Training step with dual-modal dual-view inputs.
        
        Args:
            batch: ((view1_time, view1_freq), (view2_time, view2_freq))
        """
        # Unpack batch
        (time_q, freq_q), (time_k, freq_k) = batch
        
        # Forward pass
        outputs = self(time_q, freq_q, time_k, freq_k)
        q = outputs['query']
        k = outputs['key']
        queue = outputs['queue']
        
        # Compute loss
        loss = self.criterion(q, k, queue)
        
        # Compute accuracy
        acc = compute_accuracy(q, k, queue, self.temperature)
        
        # Update queue
        self._dequeue_and_enqueue(k)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('queue_ptr', float(self.queue_ptr), on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        """
        # Unpack batch (use both views for validation)
        (time_q, freq_q), (time_k, freq_k) = batch
        
        # Forward pass
        with torch.no_grad():
            outputs = self(time_q, freq_q, time_k, freq_k)
            q = outputs['query']
            k = outputs['key']
            queue = outputs['queue']
            
            loss = self.criterion(q, k, queue)
            acc = compute_accuracy(q, k, queue, self.temperature)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Optimizer
        optimizer = torch.optim.Adam(
            self.encoder_q.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        if self.scheduler_config is not None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get('T_max', 200),
                eta_min=self.scheduler_config.get('eta_min', 1e-6)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        return optimizer
    
    def get_encoder(self) -> nn.Module:
        """
        Get query encoder for downstream tasks.
        
        Returns:
            Query encoder without projection head
        """
        return self.encoder_q.get_encoder()

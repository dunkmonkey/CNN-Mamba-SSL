"""
MoCo (Momentum Contrast) PyTorch Lightning Module for self-supervised pretraining.

Implements the MoCo v2 framework with:
- Query encoder (updated by gradients)
- Momentum encoder (updated by exponential moving average)
- Dynamic queue of negative samples
- InfoNCE contrastive loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Dict, Any, Optional
from omegaconf import DictConfig
import copy

from ..losses.info_nce import MoCoLoss, compute_accuracy


class MoCoModule(L.LightningModule):
    """
    MoCo framework for self-supervised contrastive learning.
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
            encoder: Query encoder model
            encoder_config: Configuration for building momentum encoder
            queue_size: Size of the negative sample queue
            momentum: Momentum coefficient for updating momentum encoder
            temperature: Temperature parameter for InfoNCE loss
            learning_rate: Learning rate
            weight_decay: Weight decay
            scheduler_config: Optional learning rate scheduler config
        """
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        
        # Query encoder (will be updated by gradients)
        self.encoder_q = encoder
        
        # Momentum encoder (will be updated by momentum)
        self.encoder_k = copy.deepcopy(encoder)
        
        # Freeze momentum encoder parameters
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
        Update momentum encoder using exponential moving average:
        θ_k = m * θ_k + (1 - m) * θ_q
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """
        Update the queue with new keys.
        
        Args:
            keys: New keys to add to queue, shape (batch_size, dim)
        """
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace oldest entries in queue
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[ptr:] = keys[:remaining]
            self.queue[:batch_size - remaining] = keys[remaining:]
        
        # Update pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x: torch.Tensor) -> tuple:
        """
        Batch shuffle for making use of BatchNorm in DDP.
        Returns shuffled x and the indices to unshuffle.
        """
        # Only matters for distributed training
        if not self.trainer or self.trainer.world_size == 1:
            return x, None
        
        batch_size_this = x.shape[0]
        
        # Gather from all GPUs
        x_gather = self.all_gather(x, sync_grads=False)
        batch_size_all = x_gather.shape[0] * x_gather.shape[1]
        x_gather = x_gather.reshape(batch_size_all, *x.shape[1:])
        
        # Random shuffle index
        idx_shuffle = torch.randperm(batch_size_all, device=x.device)
        
        # Broadcast to all GPUs
        torch.distributed.broadcast(idx_shuffle, src=0)
        
        # Index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)
        
        # Shuffled index for this GPU
        gpu_idx = self.trainer.global_rank
        idx_this = idx_shuffle.view(-1, batch_size_this)[gpu_idx]
        
        return x_gather[idx_this], idx_unshuffle
    
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x: torch.Tensor, idx_unshuffle: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Undo batch shuffle.
        """
        if idx_unshuffle is None or not self.trainer or self.trainer.world_size == 1:
            return x
        
        batch_size_this = x.shape[0]
        
        # Gather from all GPUs
        x_gather = self.all_gather(x, sync_grads=False)
        batch_size_all = x_gather.shape[0] * x_gather.shape[1]
        x_gather = x_gather.reshape(batch_size_all, *x.shape[1:])
        
        # Restored index for this GPU
        gpu_idx = self.trainer.global_rank
        idx_this = idx_unshuffle.view(-1, batch_size_this)[gpu_idx]
        
        return x_gather[idx_this]
    
    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MoCo.
        
        Args:
            im_q: Query images (batch_size, channels, length)
            im_k: Key images (batch_size, channels, length)
            
        Returns:
            Dictionary with queries, keys, and queue
        """
        # Compute query features
        q = self.encoder_q(im_q, return_embedding=False)  # (batch, dim)
        q = F.normalize(q, dim=1)
        
        # Compute key features with no gradient
        with torch.no_grad():
            # Update momentum encoder
            self._momentum_update_key_encoder()
            
            # Shuffle for BN in DDP
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            
            # Compute keys
            k = self.encoder_k(im_k, return_embedding=False)  # (batch, dim)
            k = F.normalize(k, dim=1)
            
            # Unshuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        
        return {'query': q, 'key': k, 'queue': self.queue.clone().detach()}
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Unpack dual views
        im_q, im_k = batch  # Each is (batch_size, 1, n_samples)
        
        # Forward pass
        outputs = self(im_q, im_k)
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
        """Validation step."""
        # For validation, we just compute loss without updating queue
        im_q = batch[0]  # Just use one view
        im_k = batch[0] if isinstance(batch, tuple) and len(batch) == 1 else batch[0]
        
        # Forward pass
        with torch.no_grad():
            outputs = self(im_q, im_k)
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
        """Configure optimizer and learning rate scheduler."""
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
        Get the query encoder for downstream tasks.
        
        Returns:
            Query encoder without projection head
        """
        return self.encoder_q.get_encoder()

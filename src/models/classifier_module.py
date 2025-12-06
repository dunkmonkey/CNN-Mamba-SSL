"""
Classifier Module for downstream fine-tuning on labeled data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Dict, Any, Optional
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC

from ..utils.metrics import MetricsTracker


class ClassifierModule(L.LightningModule):
    """
    Classification module for fine-tuning pre-trained encoders.
    
    Supports two strategies:
    1. Linear Probing: Freeze encoder, only train classification head
    2. Full Fine-tuning: Train both encoder and head (with different LRs)
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 2,
        hidden_dim: Optional[int] = None,
        freeze_encoder: bool = False,
        encoder_lr_multiplier: float = 0.01,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        scheduler_config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            encoder: Pre-trained encoder
            num_classes: Number of output classes
            hidden_dim: Optional hidden layer dimension for classification head
            freeze_encoder: Whether to freeze encoder weights (linear probing)
            encoder_lr_multiplier: LR multiplier for encoder in full fine-tuning
            learning_rate: Learning rate for classification head
            weight_decay: Weight decay
            scheduler_config: Optional learning rate scheduler config
        """
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])
        
        self.encoder = encoder
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        self.encoder_lr_multiplier = encoder_lr_multiplier
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config
        
        # Freeze encoder if linear probing
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Get encoder output dimension
        encoder_out_dim = self.encoder.get_output_dim() if hasattr(self.encoder, 'get_output_dim') else 128
        
        # Classification head
        if hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(encoder_out_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Linear(encoder_out_dim, num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        self.test_metrics = MetricsTracker()
        
        # Torchmetrics
        self.train_acc = Accuracy(task='multiclass' if num_classes > 2 else 'binary', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass' if num_classes > 2 else 'binary', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass' if num_classes > 2 else 'binary', num_classes=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, length)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Extract features
        features = self.encoder(x)
        
        # If encoder returns tuple, take first element
        if isinstance(features, tuple):
            features = features[0]
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        
        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Compute predictions
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        
        # Update metrics
        self.train_acc(preds, y)
        self.train_metrics.update(preds, y, probs[:, 1] if self.num_classes == 2 else probs)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        
        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Compute predictions
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        
        # Update metrics
        self.val_acc(preds, y)
        self.val_metrics.update(preds, y, probs[:, 1] if self.num_classes == 2 else probs)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        
        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Compute predictions
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        
        # Update metrics
        self.test_acc(preds, y)
        self.test_metrics.update(preds, y, probs[:, 1] if self.num_classes == 2 else probs)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Compute and log epoch-level metrics."""
        metrics = self.train_metrics.compute(binary=(self.num_classes == 2))
        for name, value in metrics.items():
            self.log(f'train_{name}', value)
        self.train_metrics.reset()
    
    def on_validation_epoch_end(self):
        """Compute and log validation epoch-level metrics."""
        metrics = self.val_metrics.compute(binary=(self.num_classes == 2))
        for name, value in metrics.items():
            self.log(f'val_{name}', value)
        self.val_metrics.reset()
    
    def on_test_epoch_end(self):
        """Compute and log test epoch-level metrics."""
        metrics = self.test_metrics.compute(binary=(self.num_classes == 2))
        for name, value in metrics.items():
            self.log(f'test_{name}', value)
        
        # Print detailed results
        print("\n" + "="*60)
        print("Test Results:")
        print("="*60)
        for name, value in metrics.items():
            print(f"{name:30s}: {value:.4f}")
        print("="*60)
        
        self.test_metrics.reset()
    
    def configure_optimizers(self):
        """Configure optimizer with different LRs for encoder and head."""
        if self.freeze_encoder:
            # Only optimize classifier
            parameters = self.classifier.parameters()
            optimizer = torch.optim.Adam(
                parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            # Different LRs for encoder and classifier
            parameters = [
                {
                    'params': self.encoder.parameters(),
                    'lr': self.learning_rate * self.encoder_lr_multiplier
                },
                {
                    'params': self.classifier.parameters(),
                    'lr': self.learning_rate
                }
            ]
            optimizer = torch.optim.Adam(parameters, weight_decay=self.weight_decay)
        
        # Scheduler
        if self.scheduler_config is not None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get('T_max', 100),
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

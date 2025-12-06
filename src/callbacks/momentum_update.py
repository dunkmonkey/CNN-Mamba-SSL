"""
Custom PyTorch Lightning callbacks for MoCo training.
"""

import lightning as L
from lightning.pytorch.callbacks import Callback


class MomentumUpdateCallback(Callback):
    """
    Callback to log momentum encoder update information.
    
    This is mainly for monitoring; the actual momentum update
    is handled within MoCoModule.
    """
    
    def __init__(self, momentum: float = 0.999):
        """
        Args:
            momentum: Momentum coefficient
        """
        super().__init__()
        self.momentum = momentum
    
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx
    ):
        """Log momentum update info."""
        # This can be extended to log encoder parameter statistics
        pass
    
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Log momentum coefficient at epoch start."""
        pl_module.log('momentum', self.momentum, on_step=False, on_epoch=True)

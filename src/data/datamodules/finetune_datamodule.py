"""
PyTorch Lightning DataModule for fine-tuning with labeled data.
"""

import lightning as L
from torch.utils.data import DataLoader, Subset
from typing import Optional
from omegaconf import DictConfig
import torch

from ..datasets.physionet_dataset import PhysioNetPCGDataset, split_dataset
from ..datasets.augmentations import build_augmentation_pipeline


class FinetuneDataModule(L.LightningDataModule):
    """
    DataModule for supervised fine-tuning on labeled heart sound data.
    """
    
    def __init__(
        self,
        data_root: str,
        target_sr: int = 4000,
        segment_length: float = 5.0,
        n_samples: int = 20000,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        num_classes: int = 2,
        use_augmentation: bool = True,
        augmentation_config: Optional[DictConfig] = None,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            data_root: Root directory of dataset
            target_sr: Target sampling rate
            segment_length: Segment length in seconds
            n_samples: Number of samples per segment
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Pin memory for faster GPU transfer
            num_classes: Number of classification classes
            use_augmentation: Use augmentation for training
            augmentation_config: Augmentation configuration
            use_cache: Use cached data
            cache_dir: Cache directory
            seed: Random seed
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.data_root = data_root
        self.target_sr = target_sr
        self.segment_length = segment_length
        self.n_samples = n_samples
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_classes = num_classes
        self.use_augmentation = use_augmentation
        self.augmentation_config = augmentation_config
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.seed = seed
        
        # Will be initialized in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage.
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Build augmentation pipeline (only for training)
        if self.use_augmentation and self.augmentation_config is not None:
            train_transform = build_augmentation_pipeline(
                dict(self.augmentation_config),
                sr=self.target_sr
            )
        else:
            train_transform = None
        
        # Create full dataset with labels
        full_dataset = PhysioNetPCGDataset(
            data_root=self.data_root,
            target_sr=self.target_sr,
            segment_length=self.segment_length,
            split='full',
            transform=None,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
            return_label=True  # Return labels for supervised learning
        )
        
        # Split dataset
        train_indices, val_indices, test_indices = split_dataset(
            full_dataset,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed
        )
        
        # Create subset datasets
        if stage == 'fit' or stage is None:
            # Training set with augmentation
            train_dataset_raw = Subset(full_dataset, train_indices)
            self.train_dataset = TransformSubset(train_dataset_raw, train_transform)
            
            # Validation set without augmentation
            val_dataset_raw = Subset(full_dataset, val_indices)
            self.val_dataset = TransformSubset(val_dataset_raw, None)
        
        if stage == 'test' or stage is None:
            # Test set without augmentation
            test_dataset_raw = Subset(full_dataset, test_indices)
            self.test_dataset = TransformSubset(test_dataset_raw, None)
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )


class TransformSubset:
    """
    Wrapper to apply transforms to a Subset while preserving labels.
    """
    
    def __init__(self, subset, transform):
        """
        Args:
            subset: torch.utils.data.Subset
            transform: Transform to apply to audio only
        """
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        """
        Get item, apply transform to audio, and return with label.
        
        Returns:
            (audio_tensor, label_tensor)
        """
        item = self.subset[idx]
        audio_tensor = item[0]  # Shape: (1, n_samples)
        label_tensor = item[1]   # Scalar label
        
        if self.transform is not None:
            # Convert to numpy for augmentation
            audio_np = audio_tensor.squeeze(0).numpy()
            
            # Apply transform
            audio_aug = self.transform(audio_np)
            audio_tensor = torch.FloatTensor(audio_aug).unsqueeze(0)
        
        return audio_tensor, label_tensor

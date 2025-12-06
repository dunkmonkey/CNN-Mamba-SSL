"""
PyTorch Lightning DataModule for pretraining with MoCo.

Handles data loading, splitting, and dual augmentation view generation.
"""

import lightning as L
from torch.utils.data import DataLoader, Subset
from typing import Optional
from omegaconf import DictConfig
import torch

from ..datasets.physionet_dataset import PhysioNetPCGDataset, split_dataset
from ..datasets.augmentations import build_augmentation_pipeline


class MoCoTransform:
    """
    Wrapper to generate two augmented views for MoCo contrastive learning.
    """
    
    def __init__(self, base_transform):
        """
        Args:
            base_transform: Base augmentation pipeline
        """
        self.base_transform = base_transform
    
    def __call__(self, audio):
        """
        Apply augmentation twice to generate two views.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Tuple of (view1, view2)
        """
        view1 = self.base_transform(audio)
        view2 = self.base_transform(audio)
        return view1, view2


class PretrainDataModule(L.LightningDataModule):
    """
    DataModule for MoCo pretraining on PhysioNet dataset.
    """
    
    def __init__(
        self,
        data_root: str,
        target_sr: int = 4000,
        segment_length: float = 5.0,
        n_samples: int = 20000,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        use_dual_augmentation: bool = True,
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
            persistent_workers: Keep workers alive between epochs
            use_dual_augmentation: Generate two views for MoCo
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
        self.persistent_workers = persistent_workers and num_workers > 0
        self.use_dual_augmentation = use_dual_augmentation
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
        # Build augmentation pipeline
        if self.augmentation_config is not None:
            aug_pipeline = build_augmentation_pipeline(
                dict(self.augmentation_config),
                sr=self.target_sr
            )
            
            if self.use_dual_augmentation:
                train_transform = MoCoTransform(aug_pipeline)
            else:
                train_transform = aug_pipeline
        else:
            train_transform = None
        
        # Create full dataset
        full_dataset = PhysioNetPCGDataset(
            data_root=self.data_root,
            target_sr=self.target_sr,
            segment_length=self.segment_length,
            split='full',
            transform=None,  # Will apply transform in subsets
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
            return_label=False  # No labels for pretraining
        )
        
        # Split dataset
        train_indices, val_indices, test_indices = split_dataset(
            full_dataset,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed
        )
        
        # Create subset datasets with transforms
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
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_fn if self.use_dual_augmentation else None
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )
    
    def _collate_fn(self, batch):
        """
        Custom collate function for dual augmentation views.
        
        Args:
            batch: List of samples, each containing (view1, view2)
            
        Returns:
            Tuple of (batch_view1, batch_view2)
        """
        if self.use_dual_augmentation:
            # Each sample is ((view1, view2),)
            views = [item[0] for item in batch]  # Extract tuple of views
            view1_list = [v[0] for v in views]
            view2_list = [v[1] for v in views]
            
            batch_view1 = torch.stack(view1_list)
            batch_view2 = torch.stack(view2_list)
            
            return batch_view1, batch_view2
        else:
            # Standard collation
            return torch.stack([item[0] for item in batch])


class TransformSubset:
    """
    Wrapper to apply transforms to a Subset.
    """
    
    def __init__(self, subset, transform):
        """
        Args:
            subset: torch.utils.data.Subset
            transform: Transform to apply
        """
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        """
        Get item and apply transform.
        
        Returns:
            If transform generates dual views: ((view1_tensor, view2_tensor),)
            If no transform: (audio_tensor,)
        """
        item = self.subset[idx]
        audio_tensor = item[0]  # Shape: (1, n_samples)
        
        if self.transform is not None:
            # Convert to numpy for augmentation
            audio_np = audio_tensor.squeeze(0).numpy()
            
            # Apply transform
            result = self.transform(audio_np)
            
            # Check if dual views
            if isinstance(result, tuple) and len(result) == 2:
                view1, view2 = result
                view1_tensor = torch.FloatTensor(view1).unsqueeze(0)
                view2_tensor = torch.FloatTensor(view2).unsqueeze(0)
                return ((view1_tensor, view2_tensor),)
            else:
                # Single view
                audio_tensor = torch.FloatTensor(result).unsqueeze(0)
        
        return (audio_tensor,)

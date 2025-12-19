"""
Dual-Modal DataModule for DualStreamEncoder.

Generates both time-domain audio and mel-spectrogram for dual-stream learning.
"""

import lightning as L
from torch.utils.data import DataLoader, Subset
from typing import Optional, Tuple
from omegaconf import DictConfig
import torch
import torchaudio.transforms as T

from ..datasets.physionet_dataset import PhysioNetPCGDataset
from ..datasets.augmentations import build_augmentation_pipeline
from ..utils.patient_split import PatientSplitter


class DualModalTransform:
    """
    Generates dual augmented views with both time and frequency representations.
    
    For each input audio:
    1. Apply time-domain augmentation → view1_time, view2_time
    2. Convert to mel-spectrogram → view1_freq, view2_freq
    
    Returns: (view1_time, view1_freq), (view2_time, view2_freq)
    """
    
    def __init__(
        self,
        base_transform,
        mel_transform: T.MelSpectrogram,
        apply_time_aug: bool = True,
        apply_freq_aug: bool = False,
        freq_aug_pipeline=None
    ):
        """
        Args:
            base_transform: Time-domain augmentation pipeline
            mel_transform: torchaudio.transforms.MelSpectrogram
            apply_time_aug: Whether to apply time-domain augmentation
            apply_freq_aug: Whether to apply freq-domain augmentation (SpecAugment)
            freq_aug_pipeline: Frequency-domain augmentation (e.g., SpecAugment)
        """
        self.base_transform = base_transform
        self.mel_transform = mel_transform
        self.apply_time_aug = apply_time_aug
        self.apply_freq_aug = apply_freq_aug
        self.freq_aug_pipeline = freq_aug_pipeline
    
    def __call__(self, audio: torch.Tensor) -> Tuple[Tuple, Tuple]:
        """
        Generate two dual-modal views.
        
        Args:
            audio: Input audio tensor (n_samples,) or (1, n_samples)
            
        Returns:
            ((view1_time, view1_freq), (view2_time, view2_freq))
            - view*_time: (n_samples,)
            - view*_freq: (n_mels, time_frames)
        """
        # Ensure 1D
        if audio.dim() == 2:
            audio = audio.squeeze(0)
        
        # Generate view1
        if self.apply_time_aug and self.base_transform is not None:
            view1_time = self.base_transform(audio.numpy())
            view1_time = torch.FloatTensor(view1_time)
        else:
            view1_time = audio.clone()
        
        view1_freq = self.mel_transform(view1_time)  # (n_mels, time_frames)
        
        if self.apply_freq_aug and self.freq_aug_pipeline is not None:
            view1_freq = self.freq_aug_pipeline(view1_freq)
        
        # Generate view2
        if self.apply_time_aug and self.base_transform is not None:
            view2_time = self.base_transform(audio.numpy())
            view2_time = torch.FloatTensor(view2_time)
        else:
            view2_time = audio.clone()
        
        view2_freq = self.mel_transform(view2_time)
        
        if self.apply_freq_aug and self.freq_aug_pipeline is not None:
            view2_freq = self.freq_aug_pipeline(view2_freq)
        
        return (view1_time, view1_freq), (view2_time, view2_freq)


class DualModalPretrainDataModule(L.LightningDataModule):
    """
    DataModule for dual-stream MoCo pretraining with patient-wise splitting.
    
    Returns batches of:
        - view1_time: (B, n_samples)
        - view1_freq: (B, n_mels, time_frames)
        - view2_time: (B, n_samples)
        - view2_freq: (B, n_mels, time_frames)
    """
    
    def __init__(
        self,
        data_root: str,
        target_sr: int = 2000,
        segment_length: float = 5.0,
        n_samples: int = 10000,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        # Mel-spectrogram parameters
        n_mels: int = 64,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 400,
        # Augmentation parameters
        augmentation_config: Optional[DictConfig] = None,
        apply_time_aug: bool = True,
        apply_freq_aug: bool = True,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            target_sr: 2000 Hz for PCG signals
            n_samples: 2000 * 5 = 10000 samples
            n_mels: Number of mel bands (default 64)
            n_fft: FFT size (default 512)
            hop_length: Hop length (default 128) → ~500 frames for 5s
            win_length: Window length (default 400 = 200ms)
            apply_time_aug: Apply time-domain augmentation
            apply_freq_aug: Apply frequency-domain augmentation (SpecAugment)
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
        
        # Mel-spectrogram config
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # Augmentation config
        self.augmentation_config = augmentation_config
        self.apply_time_aug = apply_time_aug
        self.apply_freq_aug = apply_freq_aug
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.seed = seed
        
        # Initialize mel transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=2.0,
            normalized=True
        )
        
        # Will be initialized in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets with patient-wise splitting.
        """
        # Build time-domain augmentation
        time_aug = None
        if self.apply_time_aug and self.augmentation_config is not None:
            time_aug = build_augmentation_pipeline(
                dict(self.augmentation_config),
                sr=self.target_sr
            )
        
        # Build frequency-domain augmentation (SpecAugment)
        freq_aug = None
        if self.apply_freq_aug:
            from ..datasets.augmentations import SpecAugmentTorch
            freq_aug = SpecAugmentTorch(
                freq_mask_param=20,
                time_mask_param=40,
                num_freq_masks=2,
                num_time_masks=2
            )
        
        # Create dual-modal transform
        dual_transform = DualModalTransform(
            base_transform=time_aug,
            mel_transform=self.mel_transform,
            apply_time_aug=self.apply_time_aug,
            apply_freq_aug=self.apply_freq_aug,
            freq_aug_pipeline=freq_aug
        )
        
        # Create full dataset
        full_dataset = PhysioNetPCGDataset(
            data_root=self.data_root,
            target_sr=self.target_sr,
            segment_length=self.segment_length,
            split='full',
            transform=None,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
            return_label=False
        )
        
        # Patient-wise split
        splitter = PatientSplitter(
            dataset=full_dataset,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            stratify=False,  # No labels for pretraining
            seed=self.seed
        )
        
        train_indices, val_indices, test_indices = splitter.split()
        
        # Create datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = DualModalSubset(
                Subset(full_dataset, train_indices),
                dual_transform
            )
            self.val_dataset = DualModalSubset(
                Subset(full_dataset, val_indices),
                dual_transform
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = DualModalSubset(
                Subset(full_dataset, test_indices),
                dual_transform
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """
        Collate dual-modal dual-view batches.
        
        Args:
            batch: List of ((view1_time, view1_freq), (view2_time, view2_freq))
        
        Returns:
            (batch_view1_time, batch_view1_freq), (batch_view2_time, batch_view2_freq)
        """
        view1_list = [item[0] for item in batch]  # List of (time, freq)
        view2_list = [item[1] for item in batch]
        
        # Stack time-domain
        batch_view1_time = torch.stack([v[0] for v in view1_list])  # (B, n_samples)
        batch_view2_time = torch.stack([v[0] for v in view2_list])
        
        # Stack frequency-domain
        batch_view1_freq = torch.stack([v[1] for v in view1_list])  # (B, n_mels, frames)
        batch_view2_freq = torch.stack([v[1] for v in view2_list])
        
        return (batch_view1_time, batch_view1_freq), (batch_view2_time, batch_view2_freq)


class DualModalSubset:
    """
    Wrapper for Subset with dual-modal transform.
    """
    
    def __init__(self, subset, transform: DualModalTransform):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        """
        Returns:
            ((view1_time, view1_freq), (view2_time, view2_freq))
        """
        item = self.subset[idx]
        audio_tensor = item[0]  # (1, n_samples) or (n_samples,)
        
        # Apply dual-modal transform
        return self.transform(audio_tensor)

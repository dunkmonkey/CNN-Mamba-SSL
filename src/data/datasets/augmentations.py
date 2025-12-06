"""
Data augmentation techniques for heart sound (PCG) signals.

Implements six key augmentation strategies:
1. Gaussian Noise - Simulates recording environment noise
2. Time Shift - Shifts signal in time domain
3. Amplitude Scaling - Simulates varying stethoscope pressure
4. Time Warping - Non-linear time stretching/compression
5. Time Masking - Masks random time segments
6. Low-pass Filter - Simulates different stethoscope frequency responses
"""

import numpy as np
import torch
from scipy import signal
from typing import Optional, Dict, Any
import random


class PCGAugmentation:
    """Base class for PCG augmentation transforms."""
    
    def __init__(self, prob: float = 0.5):
        """
        Args:
            prob: Probability of applying the augmentation
        """
        self.prob = prob
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply augmentation with given probability.
        
        Args:
            audio: Input audio signal, shape (n_samples,)
            
        Returns:
            Augmented audio signal
        """
        if random.random() < self.prob:
            return self.apply(audio)
        return audio
    
    def apply(self, audio: np.ndarray) -> np.ndarray:
        """Apply the augmentation. To be implemented by subclasses."""
        raise NotImplementedError


class GaussianNoise(PCGAugmentation):
    """Add Gaussian noise to simulate recording environment noise."""
    
    def __init__(self, prob: float = 0.5, std_range: tuple = (0.005, 0.02)):
        """
        Args:
            prob: Probability of applying noise
            std_range: Range of noise standard deviation
        """
        super().__init__(prob)
        self.std_range = std_range
    
    def apply(self, audio: np.ndarray) -> np.ndarray:
        std = np.random.uniform(*self.std_range)
        noise = np.random.normal(0, std, audio.shape)
        return audio + noise


class TimeShift(PCGAugmentation):
    """Shift signal in time domain."""
    
    def __init__(self, prob: float = 0.5, shift_range: tuple = (-0.2, 0.2)):
        """
        Args:
            prob: Probability of applying shift
            shift_range: Range of shift as fraction of signal length
        """
        super().__init__(prob)
        self.shift_range = shift_range
    
    def apply(self, audio: np.ndarray) -> np.ndarray:
        shift_frac = np.random.uniform(*self.shift_range)
        shift_samples = int(len(audio) * shift_frac)
        return np.roll(audio, shift_samples)


class AmplitudeScaling(PCGAugmentation):
    """Scale amplitude to simulate varying stethoscope contact pressure."""
    
    def __init__(self, prob: float = 0.5, scale_range: tuple = (0.8, 1.2)):
        """
        Args:
            prob: Probability of applying scaling
            scale_range: Range of scaling factors
        """
        super().__init__(prob)
        self.scale_range = scale_range
    
    def apply(self, audio: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(*self.scale_range)
        return audio * scale


class TimeWarping(PCGAugmentation):
    """
    Non-linear time stretching/compression to simulate heart rate variations.
    Uses linear interpolation for simplicity.
    """
    
    def __init__(self, prob: float = 0.3, warp_range: tuple = (0.95, 1.05)):
        """
        Args:
            prob: Probability of applying warping
            warp_range: Range of warping factors
        """
        super().__init__(prob)
        self.warp_range = warp_range
    
    def apply(self, audio: np.ndarray) -> np.ndarray:
        warp_factor = np.random.uniform(*self.warp_range)
        n_samples = len(audio)
        
        # Create warped time indices
        original_indices = np.arange(n_samples)
        warped_indices = original_indices * warp_factor
        
        # Clip to valid range and interpolate
        warped_indices = np.clip(warped_indices, 0, n_samples - 1)
        warped_audio = np.interp(original_indices, warped_indices, audio)
        
        return warped_audio


class TimeMasking(PCGAugmentation):
    """
    Mask random time segments to force model to rely on context.
    Similar to SpecAugment but in time domain.
    """
    
    def __init__(
        self, 
        prob: float = 0.3, 
        mask_ratio_range: tuple = (0.05, 0.15),
        num_masks: int = 2
    ):
        """
        Args:
            prob: Probability of applying masking
            mask_ratio_range: Range of mask length as fraction of signal length
            num_masks: Number of mask segments to apply
        """
        super().__init__(prob)
        self.mask_ratio_range = mask_ratio_range
        self.num_masks = num_masks
    
    def apply(self, audio: np.ndarray) -> np.ndarray:
        audio_masked = audio.copy()
        n_samples = len(audio)
        
        for _ in range(self.num_masks):
            # Random mask length
            mask_ratio = np.random.uniform(*self.mask_ratio_range)
            mask_length = int(n_samples * mask_ratio)
            
            # Random mask position
            mask_start = np.random.randint(0, n_samples - mask_length + 1)
            mask_end = mask_start + mask_length
            
            # Apply mask (set to zero)
            audio_masked[mask_start:mask_end] = 0
        
        return audio_masked


class LowPassFilter(PCGAugmentation):
    """
    Apply low-pass filter to simulate different stethoscope frequency responses.
    Most important augmentation for PCG according to research.
    """
    
    def __init__(
        self, 
        prob: float = 0.5, 
        cutoff_freq_range: tuple = (200, 500),
        filter_order: int = 5,
        sr: int = 4000
    ):
        """
        Args:
            prob: Probability of applying filter
            cutoff_freq_range: Range of cutoff frequencies in Hz
            filter_order: Butterworth filter order
            sr: Sampling rate in Hz
        """
        super().__init__(prob)
        self.cutoff_freq_range = cutoff_freq_range
        self.filter_order = filter_order
        self.sr = sr
    
    def apply(self, audio: np.ndarray) -> np.ndarray:
        cutoff_freq = np.random.uniform(*self.cutoff_freq_range)
        
        # Design Butterworth low-pass filter
        nyquist = self.sr / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        # Ensure cutoff is in valid range
        normalized_cutoff = np.clip(normalized_cutoff, 0.01, 0.99)
        
        b, a = signal.butter(self.filter_order, normalized_cutoff, btype='low')
        
        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio


class ComposeAugmentations:
    """
    Compose multiple augmentations with optional random selection.
    """
    
    def __init__(
        self, 
        augmentations: list,
        random_compose: bool = True,
        max_augmentations: int = 4
    ):
        """
        Args:
            augmentations: List of augmentation objects
            random_compose: If True, randomly select subset of augmentations
            max_augmentations: Maximum number of augmentations to apply
        """
        self.augmentations = augmentations
        self.random_compose = random_compose
        self.max_augmentations = max_augmentations
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Apply augmentations sequentially."""
        if self.random_compose:
            # Randomly select augmentations to apply
            num_augs = np.random.randint(1, min(len(self.augmentations), self.max_augmentations) + 1)
            selected_augs = random.sample(self.augmentations, num_augs)
        else:
            selected_augs = self.augmentations
        
        # Apply selected augmentations
        for aug in selected_augs:
            audio = aug(audio)
        
        return audio


def build_augmentation_pipeline(config: Dict[str, Any], sr: int = 4000) -> ComposeAugmentations:
    """
    Build augmentation pipeline from configuration.
    
    Args:
        config: Augmentation configuration dictionary
        sr: Sampling rate
        
    Returns:
        ComposeAugmentations object
    """
    augmentations = []
    
    # Gaussian Noise
    if config.get('gaussian_noise', {}).get('enabled', True):
        augmentations.append(GaussianNoise(
            prob=config['gaussian_noise'].get('prob', 0.5),
            std_range=tuple(config['gaussian_noise'].get('std_range', [0.005, 0.02]))
        ))
    
    # Time Shift
    if config.get('time_shift', {}).get('enabled', True):
        augmentations.append(TimeShift(
            prob=config['time_shift'].get('prob', 0.5),
            shift_range=tuple(config['time_shift'].get('shift_range', [-0.2, 0.2]))
        ))
    
    # Amplitude Scaling
    if config.get('amplitude_scaling', {}).get('enabled', True):
        augmentations.append(AmplitudeScaling(
            prob=config['amplitude_scaling'].get('prob', 0.5),
            scale_range=tuple(config['amplitude_scaling'].get('scale_range', [0.8, 1.2]))
        ))
    
    # Time Warping
    if config.get('time_warping', {}).get('enabled', True):
        augmentations.append(TimeWarping(
            prob=config['time_warping'].get('prob', 0.3),
            warp_range=tuple(config['time_warping'].get('warp_range', [0.95, 1.05]))
        ))
    
    # Time Masking
    if config.get('time_masking', {}).get('enabled', True):
        augmentations.append(TimeMasking(
            prob=config['time_masking'].get('prob', 0.3),
            mask_ratio_range=tuple(config['time_masking'].get('mask_ratio_range', [0.05, 0.15])),
            num_masks=config['time_masking'].get('num_masks', 2)
        ))
    
    # Low-pass Filter
    if config.get('lowpass_filter', {}).get('enabled', True):
        augmentations.append(LowPassFilter(
            prob=config['lowpass_filter'].get('prob', 0.5),
            cutoff_freq_range=tuple(config['lowpass_filter'].get('cutoff_freq_range', [200, 500])),
            filter_order=config['lowpass_filter'].get('filter_order', 5),
            sr=sr
        ))
    
    # Compose augmentations
    pipeline = ComposeAugmentations(
        augmentations=augmentations,
        random_compose=config.get('random_compose', True),
        max_augmentations=config.get('max_augmentations', 4)
    )
    
    return pipeline


# Convenience function for simple usage
def get_default_augmentation(sr: int = 4000) -> ComposeAugmentations:
    """Get default augmentation pipeline with recommended settings."""
    config = {
        'gaussian_noise': {'enabled': True, 'prob': 0.5, 'std_range': [0.005, 0.02]},
        'time_shift': {'enabled': True, 'prob': 0.5, 'shift_range': [-0.2, 0.2]},
        'amplitude_scaling': {'enabled': True, 'prob': 0.5, 'scale_range': [0.8, 1.2]},
        'time_warping': {'enabled': True, 'prob': 0.3, 'warp_range': [0.95, 1.05]},
        'time_masking': {'enabled': True, 'prob': 0.3, 'mask_ratio_range': [0.05, 0.15], 'num_masks': 2},
        'lowpass_filter': {'enabled': True, 'prob': 0.5, 'cutoff_freq_range': [200, 500], 'filter_order': 5},
        'random_compose': True,
        'max_augmentations': 4
    }
    return build_augmentation_pipeline(config, sr)

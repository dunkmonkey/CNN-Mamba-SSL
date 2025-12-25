"""
Audio processing utilities for heart sound signals.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional
from pathlib import Path


def load_audio(
    file_path: str,
    sr: Optional[int] = None,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and optionally resample.
    
    Args:
        file_path: Path to audio file
        sr: Target sampling rate (None to keep original)
        mono: Convert to mono
        normalize: Normalize to [-1, 1]
        
    Returns:
        Tuple of (audio_array, sampling_rate)
    """
    audio, orig_sr = librosa.load(file_path, sr=sr, mono=mono)
    
    if normalize and np.abs(audio).max() > 0:
        audio = audio / np.abs(audio).max()
    
    return audio, orig_sr if sr is None else sr


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """Resample audio to target sampling rate.

    Args:
        audio: Audio array
        orig_sr: Original sampling rate
        target_sr: Target sampling rate

    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio
    return librosa.resample(y=audio, orig_sr=orig_sr, target_sr=target_sr)


def save_audio(
    file_path: str,
    audio: np.ndarray,
    sr: int,
    subtype: str = 'PCM_16'
):
    """
    Save audio to file.
    
    Args:
        file_path: Output file path
        audio: Audio array
        sr: Sampling rate
        subtype: Audio format subtype
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(file_path, audio, sr, subtype=subtype)


def compute_energy(audio: np.ndarray) -> float:
    """
    Compute signal energy.
    
    Args:
        audio: Audio signal
        
    Returns:
        Energy value
    """
    return np.sum(audio ** 2)


def compute_zcr(audio: np.ndarray) -> float:
    """
    Compute zero-crossing rate.
    
    Args:
        audio: Audio signal
        
    Returns:
        Zero-crossing rate
    """
    return np.mean(librosa.zero_crossings(audio, pad=False))


def compute_spectral_centroid(
    audio: np.ndarray,
    sr: int
) -> np.ndarray:
    """
    Compute spectral centroid.
    
    Args:
        audio: Audio signal
        sr: Sampling rate
        
    Returns:
        Spectral centroid over time
    """
    return librosa.feature.spectral_centroid(y=audio, sr=sr)[0]


def compute_mfcc(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Compute MFCC features.
    
    Args:
        audio: Audio signal
        sr: Sampling rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length
        
    Returns:
        MFCC features of shape (n_mfcc, n_frames)
    """
    return librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 64,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Compute mel spectrogram.
    
    Args:
        audio: Audio signal
        sr: Sampling rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length
        
    Returns:
        Mel spectrogram of shape (n_mels, n_frames)
    """
    return librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )


def normalize_audio(audio: np.ndarray, method: str = 'peak') -> np.ndarray:
    """
    Normalize audio signal.
    
    Args:
        audio: Audio signal
        method: Normalization method ('peak' or 'rms')
        
    Returns:
        Normalized audio
    """
    if method == 'peak':
        if np.abs(audio).max() > 0:
            return audio / np.abs(audio).max()
    elif method == 'rms':
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            return audio / rms
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return audio


def trim_silence(
    audio: np.ndarray,
    sr: int,
    top_db: int = 40,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Trim leading and trailing silence.
    
    Args:
        audio: Audio signal
        sr: Sampling rate
        top_db: Threshold in dB
        frame_length: Frame length for energy calculation
        hop_length: Hop length
        
    Returns:
        Trimmed audio
    """
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    return trimmed


def split_into_segments(
    audio: np.ndarray,
    segment_length: int,
    hop_length: Optional[int] = None,
    pad: bool = False
) -> list:
    """
    Split audio into fixed-length segments.
    
    Args:
        audio: Audio signal
        segment_length: Length of each segment in samples
        hop_length: Hop between segments (None for non-overlapping)
        pad: Whether to pad the last segment
        
    Returns:
        List of audio segments
    """
    if hop_length is None:
        hop_length = segment_length
    
    segments = []
    start = 0
    
    while start < len(audio):
        end = start + segment_length
        segment = audio[start:end]
        
        if len(segment) == segment_length:
            segments.append(segment)
        elif pad and len(segment) > 0:
            # Pad the last segment
            padded = np.pad(segment, (0, segment_length - len(segment)), mode='constant')
            segments.append(padded)
        
        start += hop_length
    
    return segments

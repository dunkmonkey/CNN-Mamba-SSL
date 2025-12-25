"""
PhysioNet CinC Challenge 2016 Dataset loader for heart sound signals.

Handles:
- Parsing .hea header files for metadata
- Loading .wav audio files
- Resampling to target sampling rate (4000 Hz)
- Random cropping of variable-length signals
- Data augmentation integration
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple, Callable, List, Dict
import torch
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class PhysioNetPCGDataset(Dataset):
    """
    PhysioNet CinC Challenge 2016 PCG dataset.
    
    Supports both training-a directory and challenge_2016_data_set directory.
    """
    
    def __init__(
        self,
        data_root: str,
        target_sr: int = 4000,
        segment_length: float = 5.0,
        split: str = 'train',
        transform: Optional[Callable] = None,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        return_label: bool = False
    ):
        """
        Args:
            data_root: Root directory containing the dataset
            target_sr: Target sampling rate in Hz
            segment_length: Length of audio segments in seconds
            split: 'train', 'val', or 'test'
            transform: Optional transform/augmentation to apply
            use_cache: Whether to cache processed data
            cache_dir: Directory for cached data
            return_label: Whether to return labels (for supervised fine-tuning)
        """
        self.data_root = Path(data_root)
        self.target_sr = target_sr
        self.segment_length = segment_length
        self.n_samples = int(target_sr * segment_length)
        self.split = split
        self.transform = transform
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.return_label = return_label
        
        # Discover audio files
        self.file_list = self._discover_files()
        
        # Load labels if available
        self.labels = self._load_labels()
        
        # Cache setup
        if self.use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._build_cache()
    
    def _discover_files(self) -> List[Dict[str, str]]:
        """Discover all .wav files in the dataset directory."""
        file_list = []
        
        # Check for training-a directory
        training_a_dir = self.data_root / 'training-a'
        if training_a_dir.exists():
            for hea_file in sorted(training_a_dir.glob('*.hea')):
                wav_file = hea_file.with_suffix('.wav')
                if wav_file.exists():
                    file_list.append({
                        'hea_path': str(hea_file),
                        'wav_path': str(wav_file),
                        'record_id': hea_file.stem
                    })
        
        # Check for challenge_2016_data_set directory
        challenge_dir = self.data_root / 'challenge_2016_data_set'
        if challenge_dir.exists():
            # Check subdirectories 0/ and 1/
            for subdir in ['0', '1']:
                subdir_path = challenge_dir / subdir
                if subdir_path.exists():
                    for wav_file in sorted(subdir_path.glob('*.wav')):
                        file_list.append({
                            'hea_path': None,  # No .hea files in challenge set
                            'wav_path': str(wav_file),
                            'record_id': wav_file.stem,
                            'label': 0 if subdir == '0' else 1  # 0=normal, 1=abnormal
                        })
        
        if len(file_list) == 0:
            raise ValueError(f"No audio files found in {self.data_root}")
        
        print(f"Found {len(file_list)} audio files in {self.data_root}")
        return file_list
    
    def _load_labels(self) -> Optional[Dict[str, int]]:
        """Load labels from REFERENCE.csv if available."""
        reference_file = self.data_root / 'challenge_2016_data_set' / 'REFERENCE.csv'
        
        if not reference_file.exists():
            # Check if labels are in file_list (from subdirectory structure)
            if self.file_list and 'label' in self.file_list[0]:
                labels = {item['record_id']: item['label'] for item in self.file_list}
                return labels
            return None
        
        # Load from REFERENCE.csv
        df = pd.read_csv(reference_file, header=None, names=['record_id', 'label'])
        
        # Convert labels: 1=normal (0), -1=abnormal (1)
        df['label'] = df['label'].apply(lambda x: 0 if x == 1 else 1)
        
        labels = dict(zip(df['record_id'].astype(str), df['label']))
        print(f"Loaded {len(labels)} labels from REFERENCE.csv")
        
        return labels
    
    def _parse_hea_file(self, hea_path: str) -> Dict[str, any]:
        """
        Parse .hea header file to extract metadata.
        
        Format example:
        a0001 2 2000 71332
        a0001.wav 16 1000 16 0 0 0 0 PCG
        a0001.dat 16 1000 16 0 0 0 0 ECG
        # Normal
        """
        metadata = {}
        
        with open(hea_path, 'r') as f:
            lines = f.readlines()
        
        # First line: record_name n_signals sampling_frequency n_samples
        first_line = lines[0].strip().split()
        metadata['record_name'] = first_line[0]
        metadata['n_signals'] = int(first_line[1])
        metadata['sampling_frequency'] = int(first_line[2])
        metadata['n_samples'] = int(first_line[3])
        
        # Find label line (starts with #)
        for line in lines:
            if line.strip().startswith('#'):
                label_text = line.strip()[1:].strip()
                metadata['label_text'] = label_text
                # Convert to binary: Normal=0, Abnormal=1
                metadata['label'] = 0 if 'normal' in label_text.lower() else 1
                break
        
        return metadata
    
    def _load_audio(self, wav_path: str, hea_path: Optional[str] = None) -> np.ndarray:
        """
        Load and resample audio to target sampling rate.
        
        Args:
            wav_path: Path to .wav file
            hea_path: Optional path to .hea file for metadata
            
        Returns:
            Audio signal as numpy array
        """
        # Load audio
        audio, sr = librosa.load(wav_path, sr=None, mono=True)
        
        # Resample if necessary
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        
        # Normalize to [-1, 1]
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max()
        
        return audio
    
    def _random_crop(self, audio: np.ndarray) -> np.ndarray:
        """
        Randomly crop audio to fixed length, or pad if too short.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Fixed-length audio segment
        """
        if len(audio) >= self.n_samples:
            # Random crop
            start_idx = np.random.randint(0, len(audio) - self.n_samples + 1)
            return audio[start_idx:start_idx + self.n_samples]
        else:
            # Pad with zeros
            pad_length = self.n_samples - len(audio)
            return np.pad(audio, (0, pad_length), mode='constant')
    
    def _build_cache(self):
        """Pre-process and cache all audio files."""
        cache_file = self.cache_dir / f'cache_{self.split}.pkl'
        
        if cache_file.exists():
            print(f"Loading cached data from {cache_file}")
            return
        
        print(f"Building cache for {len(self.file_list)} files...")
        cached_data = []
        
        for item in tqdm(self.file_list):
            try:
                audio = self._load_audio(item['wav_path'], item['hea_path'])
                
                # Get label
                record_id = item['record_id']
                label = None
                if self.labels is not None and record_id in self.labels:
                    label = self.labels[record_id]
                elif 'label' in item:
                    label = item['label']
                
                cached_data.append({
                    'audio': audio,
                    'record_id': record_id,
                    'label': label
                })
            except Exception as e:
                print(f"Error processing {item['wav_path']}: {e}")
        
        # Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        
        print(f"Cache built and saved to {cache_file}")
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample.
        
        Returns:
            If return_label=False: (audio_tensor,)
            If return_label=True: (audio_tensor, label_tensor)
        """
        # Load from cache if available
        if self.use_cache and self.cache_dir:
            cache_file = self.cache_dir / f'cache_{self.split}.pkl'
            if cache_file.exists():
                if not hasattr(self, '_cached_data'):
                    with open(cache_file, 'rb') as f:
                        self._cached_data = pickle.load(f)

                item = self._cached_data[idx]
                audio = item['audio'].copy()

                # Always recompute label from current label mapping if available
                # to avoid stale labels stored in old cache files.
                record_id = item.get('record_id')
                label = None
                if self.labels is not None and record_id in self.labels:
                    label = self.labels[record_id]
                elif 'label' in item:
                    label = item['label']
            else:
                # Load directly
                file_info = self.file_list[idx]
                audio = self._load_audio(file_info['wav_path'], file_info['hea_path'])

                # Get label
                record_id = file_info['record_id']
                label = None
                if self.labels is not None and record_id in self.labels:
                    label = self.labels[record_id]
                elif 'label' in file_info:
                    label = file_info['label']
        else:
            # Load directly
            file_info = self.file_list[idx]
            audio = self._load_audio(file_info['wav_path'], file_info['hea_path'])

            # Get label
            record_id = file_info['record_id']
            label = None
            if self.labels is not None and record_id in self.labels:
                label = self.labels[record_id]
            elif 'label' in file_info:
                label = file_info['label']
        
        # Random crop to fixed length
        audio = self._random_crop(audio)
        
        # Apply augmentation if provided
        if self.transform is not None:
            audio = self.transform(audio)
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # Shape: (1, n_samples)
        
        if self.return_label and label is not None:
            label_tensor = torch.tensor(label, dtype=torch.long)
            return audio_tensor, label_tensor
        
        return (audio_tensor,)


def split_dataset(
    dataset: PhysioNetPCGDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify: bool = False
) -> Tuple[List[int], List[int], List[int]]:
    """Split dataset indices into train/val/test sets.

    By default this performs a random split. If ``stratify=True`` and the
    dataset provides labels, a label-stratified split is performed so that
    each split preserves the global class distribution as much as possible.

    Args:
        dataset: PhysioNetPCGDataset instance
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility
        stratify: Whether to perform label-stratified splitting

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    n_samples = len(dataset)
    indices = np.arange(n_samples)

    if not stratify:
        # Original random split behaviour
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_samples)

        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        train_indices = perm[:n_train].tolist()
        val_indices = perm[n_train:n_train + n_val].tolist()
        test_indices = perm[n_train + n_val:].tolist()
    else:
        # Label-stratified split (used for supervised fine-tuning)
        # Build label list for all indices
        all_labels: List[int] = []
        for idx in indices:
            file_info = dataset.file_list[int(idx)]
            record_id = file_info["record_id"]

            label = None
            if getattr(dataset, "labels", None) is not None and record_id in dataset.labels:
                label = dataset.labels[record_id]
            elif "label" in file_info:
                label = file_info["label"]

            if label is None:
                raise ValueError(
                    "Cannot perform stratified split because some samples have no label. "
                    "Disable stratify or ensure all samples are labeled."
                )

            all_labels.append(label)

        all_labels = np.array(all_labels)

        # First split: train vs (val+test)
        test_val_ratio = val_ratio + test_ratio
        train_idx, temp_idx, _, temp_labels = train_test_split(
            indices,
            all_labels,
            test_size=test_val_ratio,
            random_state=seed,
            stratify=all_labels
        )

        # Second split: val vs test from temp
        # Proportion of test within (val+test)
        if test_val_ratio == 0:
            raise ValueError("val_ratio + test_ratio must be > 0 for stratified split")

        test_size_within_temp = test_ratio / test_val_ratio
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx,
            temp_labels,
            test_size=test_size_within_temp,
            random_state=seed,
            stratify=temp_labels
        )

        train_indices = train_idx.tolist()
        val_indices = val_idx.tolist()
        test_indices = test_idx.tolist()

        # Optional: print label distribution for debugging
        unique, counts = np.unique(all_labels, return_counts=True)
        print("Global label distribution:", dict(zip(unique.tolist(), counts.tolist())))

    print(
        f"Dataset split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}"
    )

    return train_indices, val_indices, test_indices

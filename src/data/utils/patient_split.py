"""
Patient-wise data splitting utilities.

Ensures no data leakage by splitting at the patient level,
so all recordings from the same patient are in the same split.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import defaultdict
import hashlib


class PatientSplitter:
    """
    Patient-wise data splitter with stratification support.
    
    Ensures:
    1. All recordings from the same patient are in the same split
    2. Label distribution is preserved across splits (stratified)
    3. Reproducible splits via seed
    """
    
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        stratify: bool = True,
        dataset: Optional[Any] = None
    ):
        """
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            seed: Random seed for reproducibility
            stratify: Whether to stratify by label
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.stratify = stratify
        self.dataset = dataset
        
        self._rng = np.random.RandomState(seed)
    
    def extract_patient_id(self, record_id: str) -> str:
        """
        从记录 ID 中提取患者 ID。
        
        PhysioNet 2016 格式: a0001, a0002, ... (前缀 + 数字)
        PhysioNet 2022 格式: 可能包含 _0, _1 等后缀表示同一患者的多段录音
        自建数据集: 需要根据实际格式调整
        
        Args:
            record_id: 录音记录 ID
            
        Returns:
            患者 ID
        """
        # PhysioNet 2022: 移除录音段后缀 (如 12345_0 -> 12345)
        if '_' in record_id:
            parts = record_id.rsplit('_', 1)
            if parts[-1].isdigit():
                return parts[0]
        
        # PhysioNet 2016: 整个 record_id 就是患者 ID
        return record_id
    
    def split(
        self,
        records: Optional[List[Dict]] = None,
        patient_id_key: str = "patient_id",
        label_key: str = "label"
    ) -> Union[
        Tuple[List[Dict], List[Dict], List[Dict]],
        Tuple[List[int], List[int], List[int]]
    ]:
        """执行 patient-wise 划分。

        支持两种用法：
        1) records 模式：传入 records 列表，返回 (train_records, val_records, test_records)
        2) dataset 模式：records=None 且在 __init__ 里提供 dataset，返回 (train_indices, val_indices, test_indices)
        """

        if records is None:
            if self.dataset is None:
                raise ValueError("records is None and dataset was not provided")
            return self._split_dataset(self.dataset, label_key=label_key)

        return self._split_records(records, patient_id_key=patient_id_key, label_key=label_key)

    def _split_records(
        self,
        records: List[Dict],
        patient_id_key: str = "patient_id",
        label_key: str = "label"
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """records 模式：返回划分后的 records 列表。"""
        # 按患者分组
        patient_records = defaultdict(list)
        patient_labels = {}
        
        for record in records:
            pid = record.get(patient_id_key) or self.extract_patient_id(record.get("record_id", ""))
            patient_records[pid].append(record)
            
            # 使用第一条记录的标签作为患者标签（假设同一患者标签一致）
            if pid not in patient_labels and label_key in record:
                patient_labels[pid] = record[label_key]
        
        patient_ids = list(patient_records.keys())
        
        if self.stratify and patient_labels:
            train_pids, val_pids, test_pids = self._stratified_split(
                patient_ids, patient_labels
            )
        else:
            train_pids, val_pids, test_pids = self._random_split(patient_ids)
        
        # 收集各划分的记录
        train_records = [r for pid in train_pids for r in patient_records[pid]]
        val_records = [r for pid in val_pids for r in patient_records[pid]]
        test_records = [r for pid in test_pids for r in patient_records[pid]]
        
        # 打印统计信息
        self._print_split_stats(
            train_records, val_records, test_records,
            train_pids, val_pids, test_pids,
            label_key
        )
        
        return train_records, val_records, test_records

    def _split_dataset(
        self,
        dataset: Any,
        label_key: str = "label"
    ) -> Tuple[List[int], List[int], List[int]]:
        """dataset 模式：返回划分后的样本索引列表。"""
        # Build per-sample metadata from common dataset shapes
        samples: List[Dict[str, Any]] = []

        if hasattr(dataset, "file_list"):
            file_list = getattr(dataset, "file_list")
            for idx, item in enumerate(file_list):
                if isinstance(item, dict):
                    record_id = item.get("record_id") or Path(item.get("wav_path", str(idx))).stem
                    label = item.get(label_key, item.get("label", None))
                else:
                    record_id = Path(str(item)).stem
                    label = None
                samples.append({"index": idx, "record_id": record_id, "label": label})
        elif hasattr(dataset, "file_paths"):
            file_paths = getattr(dataset, "file_paths")
            labels = getattr(dataset, "labels", None)
            for idx, p in enumerate(file_paths):
                record_id = Path(str(p)).stem
                label = None
                if isinstance(labels, dict):
                    label = labels.get(record_id)
                samples.append({"index": idx, "record_id": record_id, "label": label})
        else:
            # Fallback: rely on __len__ only
            n = len(dataset)
            for idx in range(n):
                samples.append({"index": idx, "record_id": str(idx), "label": None})

        # Group by patient
        patient_to_indices: Dict[str, List[int]] = defaultdict(list)
        patient_labels: Dict[str, int] = {}
        for s in samples:
            pid = self.extract_patient_id(str(s.get("record_id", "")))
            patient_to_indices[pid].append(int(s["index"]))
            if self.stratify and s.get("label") is not None and pid not in patient_labels:
                try:
                    patient_labels[pid] = int(s.get("label"))
                except Exception:
                    pass

        patient_ids = list(patient_to_indices.keys())
        if self.stratify and patient_labels:
            train_pids, val_pids, test_pids = self._stratified_split(patient_ids, patient_labels)
        else:
            train_pids, val_pids, test_pids = self._random_split(patient_ids)

        train_idx = [i for pid in train_pids for i in patient_to_indices[pid]]
        val_idx = [i for pid in val_pids for i in patient_to_indices[pid]]
        test_idx = [i for pid in test_pids for i in patient_to_indices[pid]]

        return train_idx, val_idx, test_idx
    
    def _random_split(
        self,
        patient_ids: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """随机划分患者 ID。"""
        patient_ids = list(patient_ids)
        self._rng.shuffle(patient_ids)
        
        n = len(patient_ids)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        
        train_pids = patient_ids[:n_train]
        val_pids = patient_ids[n_train:n_train + n_val]
        test_pids = patient_ids[n_train + n_val:]
        
        return train_pids, val_pids, test_pids
    
    def _stratified_split(
        self,
        patient_ids: List[str],
        patient_labels: Dict[str, int]
    ) -> Tuple[List[str], List[str], List[str]]:
        """分层划分，保持标签分布。"""
        # 按标签分组患者
        label_patients = defaultdict(list)
        for pid in patient_ids:
            label = patient_labels.get(pid, -1)
            label_patients[label].append(pid)
        
        train_pids, val_pids, test_pids = [], [], []
        
        # 对每个标签类别进行划分
        for label, pids in label_patients.items():
            self._rng.shuffle(pids)
            
            n = len(pids)
            n_train = max(1, int(n * self.train_ratio))
            n_val = max(1, int(n * self.val_ratio)) if n > 2 else 0
            
            train_pids.extend(pids[:n_train])
            val_pids.extend(pids[n_train:n_train + n_val])
            test_pids.extend(pids[n_train + n_val:])
        
        return train_pids, val_pids, test_pids
    
    def _print_split_stats(
        self,
        train_records: List[Dict],
        val_records: List[Dict],
        test_records: List[Dict],
        train_pids: List[str],
        val_pids: List[str],
        test_pids: List[str],
        label_key: str
    ):
        """打印划分统计信息。"""
        print("\n" + "="*60)
        print("Patient-wise Split Statistics")
        print("="*60)
        print(f"{'Split':<10} {'Patients':<12} {'Records':<12} {'Ratio':<10}")
        print("-"*60)
        
        total_patients = len(train_pids) + len(val_pids) + len(test_pids)
        total_records = len(train_records) + len(val_records) + len(test_records)
        
        for name, pids, records in [
            ("Train", train_pids, train_records),
            ("Val", val_pids, val_records),
            ("Test", test_pids, test_records)
        ]:
            p_ratio = len(pids) / total_patients if total_patients > 0 else 0
            print(f"{name:<10} {len(pids):<12} {len(records):<12} {p_ratio:.2%}")
        
        print("-"*60)
        print(f"{'Total':<10} {total_patients:<12} {total_records:<12}")
        
        # 打印标签分布
        if train_records and label_key in train_records[0]:
            print("\nLabel Distribution:")
            print("-"*60)
            
            for name, records in [
                ("Train", train_records),
                ("Val", val_records),
                ("Test", test_records)
            ]:
                labels = [r.get(label_key, -1) for r in records]
                unique, counts = np.unique(labels, return_counts=True)
                dist = ", ".join([f"L{l}:{c}" for l, c in zip(unique, counts)])
                print(f"{name:<10} {dist}")
        
        print("="*60 + "\n")


def patient_wise_split(
    records: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify: bool = True,
    patient_id_key: str = "patient_id",
    label_key: str = "label"
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    便捷函数：执行 patient-wise 划分。
    
    Args:
        records: 记录列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        stratify: 是否分层采样
        patient_id_key: 患者 ID 键名
        label_key: 标签键名
        
    Returns:
        (train_records, val_records, test_records)
    """
    splitter = PatientSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratify=stratify
    )
    
    return splitter.split(records, patient_id_key, label_key)


def get_deterministic_split(
    patient_id: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> str:
    """
    基于患者 ID 的确定性划分（无需预先知道所有患者）。
    
    使用哈希确保同一患者始终分配到同一划分。
    适用于流式数据加载或分布式场景。
    
    Args:
        patient_id: 患者 ID
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
    Returns:
        "train", "val", 或 "test"
    """
    # 使用 MD5 哈希生成确定性随机数
    hash_value = int(hashlib.md5(patient_id.encode()).hexdigest(), 16)
    normalized = (hash_value % 10000) / 10000.0
    
    if normalized < train_ratio:
        return "train"
    elif normalized < train_ratio + val_ratio:
        return "val"
    else:
        return "test"

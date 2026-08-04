"""PyTorch dataset and ID splitting utilities."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .modalities import validate_modalities


class MultiModalDataset(Dataset):
    """Dataset holding one tensor per modality for a set of patches.

    A single index ``i`` returns, for every requested modality, the image of
    patch ``i`` -- this is what makes the samples *paired* across modalities
    and enables cross-modal contrastive learning.
    """

    def __init__(
        self,
        patches: Dict[str, np.ndarray],
        labels: np.ndarray,
        modalities: List[str],
        transforms: Optional[Dict[str, Callable[[np.ndarray], np.ndarray]]] = None,
    ) -> None:
        validate_modalities(modalities)
        self.patches = patches
        self.labels = np.asarray(labels, dtype=np.int64)
        self.modalities = list(modalities)
        self.transforms = transforms or {}
        n = self.labels.shape[0]
        for m in self.modalities:
            if m not in patches:
                raise KeyError(f"Dataset missing modality '{m}'")
            if patches[m].shape[0] != n:
                raise ValueError(
                    f"Modality '{m}' has {patches[m].shape[0]} samples, expected {n}"
                )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample: Dict[str, torch.Tensor] = {}
        for m in self.modalities:
            arr = self.patches[m][idx]
            if self.transforms and m in self.transforms:
                arr = self.transforms[m](arr)
            sample[m] = torch.from_numpy(np.ascontiguousarray(arr)).float()
        sample["label"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        sample["index"] = torch.tensor(int(idx), dtype=torch.long)
        return sample


def collate_modalities(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Default collate for the multi-modal dict samples."""
    keys = list(samples[0].keys())
    batch: Dict[str, torch.Tensor] = {}
    for k in keys:
        batch[k] = torch.stack([s[k] for s in samples])
    return batch
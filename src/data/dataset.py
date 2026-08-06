"""PyTorch dataset and ID splitting utilities."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .metadata import ImageMetadata
from .modalities import validate_modalities


class MultiModalDataset(Dataset):
    """Dataset holding one tensor per modality for a set of patches.

    A single index ``i`` returns, for every requested modality, the image of
    patch ``i`` -- this is what makes the samples *paired* across modalities
    and enables cross-modal contrastive learning.

    When ``metadata`` (a sequence of :class:`ImageMetadata`) is provided, the
    returned sample also carries ``latitude`` / ``longitude`` / ``date`` fields
    (floats, or NaN when missing) so geographic/temporal losses and evaluation
    can be applied without extra plumbing.
    """

    def __init__(
        self,
        patches: Dict[str, np.ndarray],
        labels: np.ndarray,
        modalities: List[str],
        transforms: Optional[Dict[str, Callable[[np.ndarray], np.ndarray]]] = None,
        metadata: Optional[List[ImageMetadata]] = None,
    ) -> None:
        validate_modalities(modalities)
        self.patches = patches
        self.labels = np.asarray(labels, dtype=np.int64)
        self.modalities = list(modalities)
        self.transforms = transforms or {}
        self.metadata = list(metadata) if metadata is not None else None
        n = self.labels.shape[0]
        if self.metadata is not None and len(self.metadata) != n:
            raise ValueError(f"metadata length {len(self.metadata)} != n={n}")
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
        if self.metadata is not None:
            md = self.metadata[idx]
            sample["latitude"] = torch.tensor(float(md.latitude if md.latitude is not None else float("nan")))
            sample["longitude"] = torch.tensor(float(md.longitude if md.longitude is not None else float("nan")))
            sample["date"] = torch.tensor(float(md.acquisition_date is not None), dtype=torch.float)
        return sample


def collate_modalities(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Default collate for the multi-modal dict samples."""
    keys = list(samples[0].keys())
    batch: Dict[str, torch.Tensor] = {}
    for k in keys:
        batch[k] = torch.stack([s[k] for s in samples])
    return batch
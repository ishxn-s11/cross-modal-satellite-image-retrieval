"""Unified dataset interface and factory.

All datasets -- synthetic, EuroSAT, SEN12MS (SEN1-2), So2Sat LCZ42 and
BigEarthNet-MM -- expose the same :class:`DatasetInterface` surface, so the
training pipeline, retrieval engine and web UI do not depend on which dataset
was selected. Selection happens through configuration::

    dataset:
      name: sen12ms          # synthetic | eurosat | sen12ms | so2sat | bigearthnet_mm
      root: /path/to/data
      allow_fallback: true   # fall back to synthetic when real data is absent

Real datasets are large downloads and are **never** fetched automatically. If a
real dataset is requested but its data directory is not present, the loader
raises :class:`DatasetNotFound`; the factory falls back to the fully
self-contained synthetic dataset when ``allow_fallback`` is true (default) or
re-raises with download instructions otherwise.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metadata import ImageMetadata

Patches = Dict[str, np.ndarray]  # {modality: (N, C, H, W)}


class DatasetNotFound(FileNotFoundError):
    """Raised when a requested dataset is not present on disk.

    Carries a human-readable ``hint`` with download instructions so the
    fallback path can log it before switching to synthetic data.
    """

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


class DatasetInterface(ABC):
    """Common surface implemented by every dataset backend.

    Concrete subclasses build ``patches`` ({modality: (N, C, H, W)}),
    ``labels`` (N,) and ``class_names`` and -- when available -- a parallel
    ``metadata`` list of :class:`ImageMetadata`.
    """

    name: str = "dataset"
    dataset_id: Optional[str] = None
    modalities: List[str] = ["optical"]
    sensor: Optional[str] = None          # primary sensor, e.g. "Sentinel-2"
    resolution: Optional[float] = None    # metres per pixel
    downloads_required: bool = False      # True for real remote-sensing datasets

    def __init__(
        self,
        patches: Patches,
        labels: np.ndarray,
        class_names: Sequence[str],
        metadata: Optional[Sequence[ImageMetadata]] = None,
    ) -> None:
        # The modalities actually provided by this dataset instance (subclasses
        # build ``patches`` exactly for the config-selected subset).
        self.modalities: List[str] = list(patches.keys())
        self.patches = {m: np.asarray(patches[m]) for m in self.modalities}
        self.labels = np.asarray(labels, dtype=np.int64)
        self.class_names = list(class_names)
        self._metadata = list(metadata) if metadata is not None else []
        if self._metadata and len(self._metadata) != self.n:
            raise ValueError(
                f"metadata length {len(self._metadata)} != n={self.n} for '{self.name}'"
            )
        # {modality: sensor string} used for result rendering / DB rows.
        self.modality_sensor: Dict[str, str] = {m: self.sensor or "" for m in self.modalities}

    # -- basic facts ---------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.labels.shape[0])

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def metadata(self) -> List[ImageMetadata]:
        return self._metadata

    def has_metadata(self) -> bool:
        return len(self._metadata) == self.n

    def metadata_for(self, image_id: int) -> ImageMetadata:
        """Return metadata for an image id (or an all-None record)."""
        if self.has_metadata() and 0 <= int(image_id) < self.n:
            return self._metadata[int(image_id)]
        return ImageMetadata(image_id=int(image_id), dataset=self.dataset_id)

    def to_patches(self) -> Tuple[Patches, np.ndarray, List[str]]:
        """Backward-compatible projection used by ``prepare_dataset``."""
        return self.patches, self.labels, self.class_names

    def bands(self, modality: str) -> int:
        arr = self.patches.get(modality)
        return int(arr.shape[1]) if arr is not None else 0

    # -- loading -------------------------------------------------------------
    @classmethod
    @abstractmethod
    def load(cls, cfg: Dict[str, Any], logger=None) -> "DatasetInterface":
        """Load (or generate) the dataset from configuration.

        Real datasets MUST raise :class:`DatasetNotFound` with a helpful hint
        when their data directory is absent, rather than downloading anything.
        """
        raise NotImplementedError


def _logger(logger=None):
    return logger if logger is not None else type("_L", (), {"info": lambda self, m: print(m)})()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, type] = {}


def register_dataset(name: str) -> Any:
    """Decorator registering a dataset class under one or more names."""

    def deco(cls):
        for n in (name,) if isinstance(name, str) else name:
            _REGISTRY[n] = cls
        return cls

    return deco


def available_datasets() -> List[str]:
    return sorted(_REGISTRY)


def dataset_registry() -> Dict[str, type]:
    return dict(_REGISTRY)


def resolve_dataset_name(cfg: Dict[str, Any]) -> str:
    """``dataset.name`` wins; falls back to the legacy ``dataset.source``."""
    ds = cfg.get("dataset", {})
    return str(ds.get("name") or ds.get("source") or "synthetic")


def build_dataset(cfg: Dict[str, Any], logger=None) -> DatasetInterface:
    """Construct the dataset selected by ``cfg['dataset']['name']``.

    Real datasets that are not present on disk raise :class:`DatasetNotFound`.
    When ``dataset.allow_fallback`` is true (default) the factory logs the
    download hint and returns the self-contained synthetic dataset instead.
    """
    log = _logger(logger)
    ds_cfg = cfg.get("dataset", {})
    name = resolve_dataset_name(cfg)
    allow_fallback = bool(ds_cfg.get("allow_fallback", True))

    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown dataset name '{name}'; choose from {available_datasets()}"
        )
    try:
        dataset = cls.load(cfg, logger)
    except DatasetNotFound as exc:
        if exc.hint:
            log.info(f"[data] dataset '{name}' not found:\n{exc.hint}")
        if allow_fallback:
            log.info("[data] falling back to the self-contained 'synthetic' dataset")
            return _REGISTRY["synthetic"].load(cfg, logger)
        raise
    log.info(
        f"[data] dataset={dataset.dataset_id} name={dataset.name} "
        f"N={dataset.n} classes={dataset.n_classes} modalities={dataset.modalities} "
        f"sensor={dataset.sensor} has_metadata={dataset.has_metadata()}"
    )
    return dataset


def dataset_requires_download(name: str) -> bool:
    cls = _REGISTRY.get(name)
    return bool(cls and getattr(cls, "downloads_required", False))

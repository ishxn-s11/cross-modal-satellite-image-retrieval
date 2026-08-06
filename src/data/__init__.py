"""Data package: modalities, datasets, preprocessing and metadata.

Importing this package registers every dataset backend in
:data:`src.data.interface._REGISTRY` so ``build_dataset(cfg)`` can resolve
``dataset.name`` to a concrete loader.
"""

from . import bigearthnet_mm, eurosat, sen12, so2sat, synthetic  # noqa: F401 (register backends)
from .dataset import MultiModalDataset, collate_modalities
from .interface import (
    DatasetInterface,
    DatasetNotFound,
    available_datasets,
    build_dataset,
    dataset_registry,
    register_dataset,
    resolve_dataset_name,
)
from .metadata import ImageMetadata
from .modalities import (
    DEFAULT_MODALITIES,
    MODALITIES,
    available_modalities,
    modality_channels,
    modality_nbands,
    validate_modalities,
)

__all__ = [
    "MultiModalDataset",
    "collate_modalities",
    "DatasetInterface",
    "DatasetNotFound",
    "available_datasets",
    "build_dataset",
    "dataset_registry",
    "register_dataset",
    "resolve_dataset_name",
    "ImageMetadata",
    "DEFAULT_MODALITIES",
    "MODALITIES",
    "available_modalities",
    "modality_channels",
    "modality_nbands",
    "validate_modalities",
]

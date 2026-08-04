"""Modality definitions.

Each modality is a distinct sensor view of the same scene. Samples across
modalities are *paired* (same geographic patch), which lets the system learn a
common embedding space where semantically-similar scenes are close together
irrespective of the sensing modality.

Modalities
----------
optical          : 3-band RGB composite (like a true-colour Sentinel-2 / Landsat view).
multispectral    : 8 spectral bands (Blue .. SWIR1) covering visible, red-edge, NIR, SWIR.
sar              : single-channel synthetic aperture radar intensity (VV-like),
                   sensitive to structure and roughness, not colour.
"""

from __future__ import annotations

from typing import Dict, List

MODALITIES: Dict[str, Dict[str, object]] = {
    "optical": {
        "bands": 3,
        "channels": ["Red", "Green", "Blue"],
        "short": "OPT",
        "description": "RGB optical composite",
    },
    "multispectral": {
        "bands": 8,
        "channels": [
            "Blue",
            "Green",
            "Red",
            "RedEdge1",
            "RedEdge2",
            "NIR1",
            "NIR2",
            "SWIR1",
        ],
        "short": "MS",
        "description": "8-band multispectral (visible + red-edge + NIR + SWIR)",
    },
    "sar": {
        "bands": 1,
        "channels": ["VV"],
        "short": "SAR",
        "description": "single-channel SAR intensity",
    },
}

DEFAULT_MODALITIES: List[str] = ["optical", "multispectral", "sar"]

# Column index of the Red / Green / Blue bands inside the multispectral stack.
MS_RGB_INDEX = {"R": 2, "G": 1, "B": 0}


def modality_channels(modality: str) -> List[str]:
    return list(MODALITIES[modality]["channels"])


def modality_nbands(modality: str) -> int:
    return int(MODALITIES[modality]["bands"])


def available_modalities() -> List[str]:
    return list(MODALITIES.keys())


def validate_modalities(modalities: List[str]) -> None:
    known = set(available_modalities())
    unknown = set(modalities) - known
    if unknown:
        raise ValueError(f"Unknown modality(ies): {sorted(unknown)}; known: {sorted(known)}")

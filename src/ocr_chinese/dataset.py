from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def polygons_to_mask(image_shape: tuple[int, int], polygons: list[list[list[float]]]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.int32)
        if points.shape[0] >= 3:
            cv2.fillPoly(mask, [points], color=255)
    return mask


def save_mask_from_annotation(image_path: Path, polygons: list[list[list[float]]], output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    mask = polygons_to_mask(image.shape, polygons)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)

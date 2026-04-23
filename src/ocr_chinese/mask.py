from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .detect import TextProposal


@dataclass
class MaskConfig:
    dilate_kernel: int = 3
    erode_kernel: int = 1
    min_component_area: int = 20


def rasterize_polygons(
    image_shape: tuple[int, int], proposals: Sequence[TextProposal]
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    for proposal in proposals:
        polygon = proposal.polygon.astype(np.int32)
        cv2.fillPoly(mask, [polygon], color=255)
    return mask


def postprocess_mask(mask: np.ndarray, config: MaskConfig) -> np.ndarray:
    processed = mask.copy()
    if config.dilate_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (config.dilate_kernel, config.dilate_kernel)
        )
        processed = cv2.dilate(processed, kernel, iterations=1)
    if config.erode_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (config.erode_kernel, config.erode_kernel)
        )
        processed = cv2.erode(processed, kernel, iterations=1)
    return remove_small_components(processed, config.min_component_area)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros_like(mask)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            output[labels == label_id] = 255
    return output


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)


def draw_overlay(image_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    overlay = image_bgr.copy()
    overlay[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0.0)

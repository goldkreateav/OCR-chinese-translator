from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import joblib
import numpy as np

from .detect import TextProposal


@dataclass
class FilterConfig:
    min_score: float = 0.1
    classifier_path: str | None = None
    classifier_threshold: float = 0.5
    min_area: float = 12.0
    max_aspect_ratio: float = 25.0


class TextCandidateFilter:
    """
    Lightweight filter that can use either:
    - geometry-only heuristics
    - a trained sklearn classifier on top of hand-crafted features
    """

    def __init__(self, config: FilterConfig):
        self.config = config
        self._model = None
        if config.classifier_path:
            classifier_file = Path(config.classifier_path)
            if classifier_file.exists():
                self._model = joblib.load(classifier_file)

    def keep(self, proposal: TextProposal, image_gray: np.ndarray) -> bool:
        if proposal.score < self.config.min_score:
            return False

        features = extract_features(image_gray, proposal.polygon)
        area, aspect_ratio = features[0], features[1]
        if area < self.config.min_area:
            return False
        if aspect_ratio > self.config.max_aspect_ratio:
            return False

        if self._model is not None:
            probability = float(self._model.predict_proba([features])[0][1])
            return probability >= self.config.classifier_threshold
        return True

    def filter(
        self, proposals: Sequence[TextProposal], image_gray: np.ndarray
    ) -> list[TextProposal]:
        return [proposal for proposal in proposals if self.keep(proposal, image_gray)]


def extract_features(image_gray: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    polygon_int = polygon.astype(np.int32)
    area = abs(cv2.contourArea(polygon))
    rect = cv2.minAreaRect(polygon.astype(np.float32))
    width, height = rect[1]
    short = max(1.0, min(width, height))
    long = max(width, height, 1.0)
    aspect_ratio = long / short

    x, y, w, h = cv2.boundingRect(polygon_int)
    h_img, w_img = image_gray.shape
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    roi = image_gray[y : y + h, x : x + w]

    if roi.size == 0:
        return np.array([area, aspect_ratio, 0.0, 0.0, 0.0], dtype=np.float32)

    edges = cv2.Canny(roi, 80, 160)
    edge_density = float(np.count_nonzero(edges)) / float(roi.size)
    mean_intensity = float(np.mean(roi)) / 255.0
    std_intensity = float(np.std(roi)) / 255.0

    return np.array(
        [area, aspect_ratio, edge_density, mean_intensity, std_intensity],
        dtype=np.float32,
    )

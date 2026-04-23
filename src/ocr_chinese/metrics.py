from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

import cv2
import numpy as np


@dataclass
class PixelMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    recall: float
    precision: float
    f1: float
    false_positive_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_pixel_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> PixelMetrics:
    pred = pred_mask > 0
    gt = gt_mask > 0
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())

    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    f1 = (2 * precision * recall) / max(1e-12, precision + recall)
    fpr = fp / max(1, fp + tn)

    return PixelMetrics(
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        recall=recall,
        precision=precision,
        f1=f1,
        false_positive_rate=fpr,
    )


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    return mask


def save_metrics_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

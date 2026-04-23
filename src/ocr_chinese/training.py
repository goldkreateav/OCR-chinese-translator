from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


@dataclass
class FilterTrainingConfig:
    n_estimators: int = 300
    random_state: int = 42
    class_weight: str = "balanced_subsample"


def train_filter_classifier(
    features_path: Path,
    model_output: Path,
    config: FilterTrainingConfig,
) -> dict:
    data = json.loads(features_path.read_text(encoding="utf-8"))
    x = np.asarray(data["features"], dtype=np.float32)
    y = np.asarray(data["labels"], dtype=np.int32)
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("Invalid feature dataset format.")
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        random_state=config.random_state,
        class_weight=config.class_weight,
        n_jobs=-1,
    )
    model.fit(x, y)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output)
    train_score = float(model.score(x, y))
    return {"samples": int(len(y)), "train_accuracy": train_score, "model_path": str(model_output)}


def calibrate_threshold(
    probabilities: Iterable[float],
    labels: Iterable[int],
    target_recall: float = 1.0,
) -> dict:
    probs = np.asarray(list(probabilities), dtype=np.float32)
    y = np.asarray(list(labels), dtype=np.int32)
    if len(probs) != len(y):
        raise ValueError("Probabilities and labels must have same length.")
    if len(y) == 0:
        raise ValueError("Empty calibration input.")

    best = {"threshold": 0.0, "recall": 0.0, "false_positive_rate": 1.0}
    negatives = max(1, int((y == 0).sum()))
    positives = max(1, int((y == 1).sum()))

    for threshold in np.linspace(0.0, 1.0, 201):
        pred = probs >= threshold
        tp = int(np.logical_and(pred, y == 1).sum())
        fp = int(np.logical_and(pred, y == 0).sum())
        fn = int(np.logical_and(np.logical_not(pred), y == 1).sum())
        recall = tp / positives
        fpr = fp / negatives

        if recall >= target_recall and fpr < best["false_positive_rate"]:
            best = {"threshold": float(threshold), "recall": float(recall), "false_positive_rate": float(fpr)}
        elif best["recall"] < target_recall and recall > best["recall"]:
            best = {"threshold": float(threshold), "recall": float(recall), "false_positive_rate": float(fpr)}
            if fn == 0:
                continue
    return best


def write_paddle_finetune_recipe(output_file: Path, dataset_root: Path) -> Path:
    """
    Writes practical fine-tune instructions for PaddleOCR DB detector.
    The actual training is expected to be run with PaddleOCR tooling.
    """
    content = {
        "framework": "PaddleOCR",
        "detector": "DBNet",
        "dataset_root": str(dataset_root.resolve()),
        "steps": [
            "Prepare det labels in PaddleOCR format: image_path\\t[{\"points\": [[x,y],...], \"transcription\": \"###\"}]",
            "Create train/val split files from manifest.",
            "Start from chinese det pretrained model and fine-tune with low threshold for max recall.",
            "Export inference model and place it under models/detector/.",
        ],
        "note": "This project provides inference + filter calibration; detector fine-tune is delegated to PaddleOCR training scripts.",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(content, indent=2), encoding="utf-8")
    return output_file

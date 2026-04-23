from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ocr_chinese.filter import extract_features


def iou_polygon_with_mask(polygon: np.ndarray, mask: np.ndarray) -> float:
    poly_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(poly_mask, [polygon.astype(np.int32)], 255)
    inter = np.logical_and(poly_mask > 0, mask > 0).sum()
    union = np.logical_or(poly_mask > 0, mask > 0).sum()
    return float(inter / max(1, union))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals-dir", type=Path, required=True)
    parser.add_argument("--gt-masks-dir", type=Path, required=True)
    parser.add_argument("--rendered-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--positive-iou", type=float, default=0.3)
    args = parser.parse_args()

    features: list[list[float]] = []
    labels: list[int] = []
    source: list[dict] = []

    for proposal_file in sorted(args.proposals_dir.glob("*_proposals.json")):
        page_id = proposal_file.stem.replace("_proposals", "")
        image = cv2.imread(str(args.rendered_dir / f"{page_id}.png"), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(args.gt_masks_dir / f"{page_id}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or gt is None:
            continue

        payload = json.loads(proposal_file.read_text(encoding="utf-8"))
        for idx, item in enumerate(payload.get("proposals", [])):
            polygon = np.asarray(item["polygon"], dtype=np.float32)
            feat = extract_features(image, polygon)
            iou = iou_polygon_with_mask(polygon, gt)
            label = 1 if iou >= args.positive_iou else 0
            features.append(feat.tolist())
            labels.append(label)
            source.append({"page_id": page_id, "proposal_idx": idx, "iou": iou})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"features": features, "labels": labels, "source": source}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(labels)} samples to {args.out}")


if __name__ == "__main__":
    main()

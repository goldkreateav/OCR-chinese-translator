from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random


@dataclass
class SplitConfig:
    dev_ratio: float = 0.5
    seed: int = 42


def prepare_labeling_manifest(
    rendered_dir: Path,
    output_manifest: Path,
    split_config: SplitConfig,
) -> dict:
    page_paths = sorted(rendered_dir.glob("page_*.png"))
    if not page_paths:
        raise FileNotFoundError(f"No rendered pages in {rendered_dir}")

    random.seed(split_config.seed)
    shuffled = page_paths[:]
    random.shuffle(shuffled)
    dev_count = int(len(shuffled) * split_config.dev_ratio)
    dev_set = set(shuffled[:dev_count])

    items = []
    for path in page_paths:
        split = "dev" if path in dev_set else "test"
        items.append(
            {
                "page_id": path.stem,
                "image_path": str(path.resolve()),
                "split": split,
                "annotations": [],
            }
        )

    manifest = {
        "format": "ocr_chinese_polygon_v1",
        "description": "Polygon annotations for chinese text masking on engineering drawings.",
        "items": items,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def export_cvat_tasks_stub(manifest_path: Path, output_dir: Path) -> Path:
    """
    Generates a helper file listing image paths.
    This file can be imported into CVAT/Label Studio workflows.
    """
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    lines = [item["image_path"] for item in data.get("items", [])]
    output_dir.mkdir(parents=True, exist_ok=True)
    index_file = output_dir / "images_for_annotation.txt"
    index_file.write_text("\n".join(lines), encoding="utf-8")
    return index_file

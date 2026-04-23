from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import statistics


NOT_FOUND_TEXT = "Текст не найден"


def load_regions(regions_dir: Path) -> list[dict]:
    items: list[dict] = []
    for file in sorted(regions_dir.glob("page_*_regions.json")):
        payload = json.loads(file.read_text(encoding="utf-8"))
        items.extend(payload.get("regions", []))
    return items


def summarize(regions: list[dict]) -> dict:
    total = len(regions)
    found = 0
    conf_values: list[float] = []
    useful_lengths: list[int] = []
    variant_counter: collections.Counter[str] = collections.Counter()
    green_count = 0
    orange_count = 0
    red_count = 0
    black_count = 0
    not_found_count = 0

    for region in regions:
        text = str(region.get("text", "")).strip()
        confidence = float(region.get("ocr_confidence", region.get("confidence", 0.0)))
        variant = str(region.get("ocr_variant", "unknown"))
        variant_counter[variant] += 1

        if confidence <= 0.0:
            black_count += 1
        elif confidence < 0.4:
            red_count += 1
        elif confidence < 0.7:
            orange_count += 1
        else:
            green_count += 1

        if text and text != NOT_FOUND_TEXT:
            found += 1
            conf_values.append(confidence)
            useful_lengths.append(len("".join(text.split())))
        else:
            not_found_count += 1

    found_ratio = found / max(1, total)
    return {
        "total_regions": total,
        "recognized_regions": found,
        "recognized_ratio": found_ratio,
        "mean_confidence": statistics.fmean(conf_values) if conf_values else 0.0,
        "mean_useful_length": statistics.fmean(useful_lengths) if useful_lengths else 0.0,
        "green_ratio": green_count / max(1, total),
        "orange_ratio": orange_count / max(1, total),
        "red_ratio": red_count / max(1, total),
        "black_ratio": black_count / max(1, total),
        "not_found_ratio": not_found_count / max(1, total),
        "variant_distribution": dict(variant_counter),
    }


def print_summary(title: str, summary: dict) -> None:
    print(f"\n{title}")
    print(f"  total_regions      : {summary['total_regions']}")
    print(f"  recognized_regions : {summary['recognized_regions']}")
    print(f"  recognized_ratio   : {summary['recognized_ratio']:.4f}")
    print(f"  mean_confidence    : {summary['mean_confidence']:.4f}")
    print(f"  mean_useful_length : {summary['mean_useful_length']:.2f}")
    print(f"  green_ratio        : {summary['green_ratio']:.4f}")
    print(f"  orange_ratio       : {summary['orange_ratio']:.4f}")
    print(f"  red_ratio          : {summary['red_ratio']:.4f}")
    print(f"  black_ratio        : {summary['black_ratio']:.4f}")
    print(f"  not_found_ratio    : {summary['not_found_ratio']:.4f}")
    variant_distribution = summary.get("variant_distribution", {})
    if variant_distribution:
        sorted_variants = sorted(
            variant_distribution.items(), key=lambda item: item[1], reverse=True
        )
        print("  variant_distribution:")
        for name, count in sorted_variants[:12]:
            print(f"    - {name}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare OCR region recognition quality before/after changes."
    )
    parser.add_argument("--regions-dir", type=Path, required=True, help="Current regions directory.")
    parser.add_argument(
        "--baseline-regions-dir",
        type=Path,
        default=None,
        help="Optional baseline regions directory for before/after comparison.",
    )
    args = parser.parse_args()

    current = load_regions(args.regions_dir)
    current_summary = summarize(current)
    print_summary("Current", current_summary)

    if args.baseline_regions_dir:
        baseline = load_regions(args.baseline_regions_dir)
        baseline_summary = summarize(baseline)
        print_summary("Baseline", baseline_summary)
        print("\nDelta")
        print(
            f"  recognized_ratio   : {current_summary['recognized_ratio'] - baseline_summary['recognized_ratio']:+.4f}"
        )
        print(
            f"  mean_confidence    : {current_summary['mean_confidence'] - baseline_summary['mean_confidence']:+.4f}"
        )
        print(
            f"  mean_useful_length : {current_summary['mean_useful_length'] - baseline_summary['mean_useful_length']:+.2f}"
        )
        print(
            f"  green_ratio        : {current_summary['green_ratio'] - baseline_summary['green_ratio']:+.4f}"
        )
        print(
            f"  not_found_ratio    : {current_summary['not_found_ratio'] - baseline_summary['not_found_ratio']:+.4f}"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

from PIL import Image

from crosswalk_detector.urban_vision import build_urban_road_density_mask, build_urban_road_line_mask, build_urban_road_seeded_mask


class Box(NamedTuple):
    x0: int
    y0: int
    x1: int
    y1: int


SAMPLES: dict[str, dict[str, list[Box]]] = {
    "winterthur-center": {
        "urban": [Box(150, 155, 355, 295), Box(15, 115, 155, 250), Box(350, 165, 505, 275)],
        "forest": [Box(170, 0, 330, 105), Box(215, 325, 370, 475)],
        "terrain": [Box(0, 330, 120, 500)],
    },
    "zurich-center": {
        "urban": [Box(0, 100, 245, 420), Box(250, 165, 400, 430), Box(330, 250, 510, 500)],
        "forest": [Box(350, 0, 505, 120), Box(410, 90, 505, 210)],
        "terrain": [Box(0, 0, 110, 80)],
    },
    "chur-center": {
        "urban": [Box(120, 80, 345, 340), Box(55, 135, 170, 360)],
        "forest": [Box(330, 150, 510, 410), Box(120, 360, 300, 510)],
        "terrain": [Box(0, 0, 130, 95), Box(335, 0, 510, 130)],
    },
}


def _box_mean(mask, box: Box) -> float:
    return float(mask[box.y0 : box.y1, box.x0 : box.x1].mean())


def _group_stats(mask, groups: dict[str, list[Box]]) -> dict[str, dict]:
    stats = {}
    for group, boxes in groups.items():
        values = [_box_mean(mask, box) for box in boxes]
        stats[group] = {
            "mean": sum(values) / len(values),
            "boxes": values,
        }
    return stats


def _confusion(mask, groups: dict[str, list[Box]]) -> dict[str, float | int]:
    true_positive = false_negative = true_negative = false_positive = 0
    for group, boxes in groups.items():
        expected = group == "urban"
        for box in boxes:
            region = mask[box.y0 : box.y1, box.x0 : box.x1]
            positives = int(region.sum())
            total = int(region.size)
            if expected:
                true_positive += positives
                false_negative += total - positives
            else:
                false_positive += positives
                true_negative += total - positives
    total = true_positive + false_negative + true_negative + false_positive
    return {
        "accuracy": (true_positive + true_negative) / total,
        "precision": true_positive / max(1, true_positive + false_positive),
        "recall": true_positive / max(1, true_positive + false_negative),
        "falsePositiveRate": false_positive / max(1, false_positive + true_negative),
        "truePositive": true_positive,
        "falseNegative": false_negative,
        "trueNegative": true_negative,
        "falsePositive": false_positive,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/urban-extent-poc-small"))
    parser.add_argument("--output", type=Path, default=Path("validation-output/urban-extent-poc-road-mask/eval.json"))
    parser.add_argument("--report", type=Path, default=Path("validation-output/urban-extent-poc-road-mask/eval-report.md"))
    args = parser.parse_args()

    results = []
    for scene_id, groups in SAMPLES.items():
        image = Image.open(args.cache_dir / f"{scene_id}-8000m-512px.jpg").convert("RGB")
        masks = {
            "roadLine": build_urban_road_line_mask(image, max_width=512, threshold=0.52),
            "roadSeededCity": build_urban_road_seeded_mask(image, max_width=512, threshold=0.52),
            "roadDensityCity": build_urban_road_density_mask(image, max_width=512, threshold=0.52),
        }
        scene_result = {"scene": scene_id, "masks": {}}
        for name, result in masks.items():
            scene_result["masks"][name] = {
                "threshold": result.threshold,
                "coverage": float(result.mask.mean()),
                "groups": _group_stats(result.mask, groups),
                "confusion": _confusion(result.mask, groups),
            }
        results.append(scene_result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.report.write_text(_build_report(results, args.output), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": results}, indent=2))


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _build_report(results: list[dict], json_path: Path) -> str:
    lines = [
        "# Urban Mask Evaluation",
        "",
        f"Raw metrics: `{json_path}`",
        "",
        "This is a small proxy evaluation over fixed image regions. It is useful for iteration, but it is not a replacement for a real labeled validation set.",
        "",
        "## Current Status",
        "",
        "| Scene | Mask | Accuracy | Precision | Recall | False positive rate |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for scene in results:
        for mask_name, mask_result in scene["masks"].items():
            confusion = mask_result["confusion"]
            lines.append(
                f"| {scene['scene']} | {mask_name} | {_pct(confusion['accuracy'])} | {_pct(confusion['precision'])} | {_pct(confusion['recall'])} | {_pct(confusion['falsePositiveRate'])} |"
            )

    lines.extend(
        [
            "",
            "## Completion Checklist",
            "",
            "- City outlines are rendered without opaque fill: covered by the latest visual sheets.",
            "- Road-like structures are highlighted as vein overlays: covered by the road detail sheets.",
            "- Winterthur forest false positive is reduced compared with the original filled city blob: covered by the road-seeded and road-line panels.",
            "- 95% accuracy target: not covered. The proxy metrics are below 95%, and there is no real labeled validation set yet.",
            "",
            "## Main Failure Modes",
            "",
            "- The road-line mask has good precision but low recall because it only catches the strongest visible road structures.",
            "- The broader road-seeded city mask covers more city area, but terrain and river structures still create false positives in difficult scenes.",
            "- Chur remains the hardest example because roads, rivers, and mountain terrain have similar thin bright structures.",
            "",
            "## Next High-Accuracy Path",
            "",
            "Use the current satellite heuristic as a proposal generator, then verify it against real labels or an authoritative road/building source. A pure heuristic can keep improving, but it cannot honestly prove 95% accuracy without ground truth.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()

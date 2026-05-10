from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from crosswalk_detector.urban_vision import _dilate, build_urban_road_line_mask


def _official_road_mask(roads_image: Image.Image) -> np.ndarray:
    rgba = roads_image.convert("RGBA")
    pixels = np.asarray(rgba, dtype=np.uint8)
    alpha = pixels[:, :, 3] > 8
    rgb = pixels[:, :, :3].astype(np.int16)
    dark_or_colored = (rgb.max(axis=2) - rgb.min(axis=2) > 12) | (rgb.mean(axis=2) < 245)
    return alpha & dark_or_colored


def _metrics(predicted: np.ndarray, target: np.ndarray, tolerance: int) -> dict[str, float | int]:
    predicted_match = predicted & _dilate(target, iterations=tolerance)
    target_match = target & _dilate(predicted, iterations=tolerance)

    true_positive = int(predicted_match.sum())
    false_positive = int((predicted & ~_dilate(target, iterations=tolerance)).sum())
    false_negative = int((target & ~_dilate(predicted, iterations=tolerance)).sum())
    true_negative = int((~predicted & ~target).sum())
    return {
        "precisionWithTolerance": true_positive / max(1, true_positive + false_positive),
        "recallWithTolerance": target_match.sum() / max(1, target.sum()),
        "f1WithTolerance": (2 * true_positive) / max(1, 2 * true_positive + false_positive + false_negative),
        "pixelAccuracyNoTolerance": (int((predicted == target).sum())) / predicted.size,
        "predictedCoverage": float(predicted.mean()),
        "targetCoverage": float(target.mean()),
        "truePositive": true_positive,
        "falsePositive": false_positive,
        "falseNegative": false_negative,
        "trueNegative": true_negative,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--satellite-dir", type=Path, default=Path("data/cache/urban-extent-poc-small"))
    parser.add_argument("--official-dir", type=Path, default=Path("data/cache/swisstopo-road-mask-poc"))
    parser.add_argument("--output", type=Path, default=Path("validation-output/swisstopo-road-mask-poc/image-vs-official-eval.json"))
    parser.add_argument("--cities", default="zurich-center,winterthur-center,chur-center")
    parser.add_argument("--size-m", type=int, default=8000)
    parser.add_argument("--image-px", type=int, default=512)
    parser.add_argument("--tolerance", type=int, default=3)
    args = parser.parse_args()

    results = []
    for scene_id in [city for city in args.cities.split(",") if city]:
        satellite = Image.open(args.satellite_dir / f"{scene_id}-{args.size_m}m-{args.image_px}px.jpg").convert("RGB")
        predicted = build_urban_road_line_mask(satellite, max_width=args.image_px, threshold=0.52).mask
        official_image = Image.open(args.official_dir / f"{scene_id}-ch.swisstopo.swisstlm3d-strassen-{args.size_m}m-{args.image_px}px.png")
        target = _official_road_mask(official_image)
        results.append({"scene": scene_id, "metrics": _metrics(predicted, target, args.tolerance)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": results}, indent=2))


if __name__ == "__main__":
    main()

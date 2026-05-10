from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from crosswalk_detector.autopilot import EARTH_RADIUS_M, GeoReference
from crosswalk_detector.urban_vision import build_urban_mask, detect_urban_contours


def _mercator_to_lon_lat(x: float, y: float) -> tuple[float, float]:
    lon = math.degrees(x / EARTH_RADIUS_M)
    lat = math.degrees(2 * math.atan(math.exp(y / EARTH_RADIUS_M)) - math.pi / 2)
    return lon, lat


def _point_to_pixel(x: float, y: float, width: int, height: int, georef: GeoReference) -> tuple[int, int]:
    lon, lat = _mercator_to_lon_lat(x, y)
    px = round((lon - georef.west) / (georef.east - georef.west) * width)
    py = round((georef.north - lat) / (georef.north - georef.south) * height)
    return px, py


def _render_debug(image_path: Path, output_dir: Path, max_width: int) -> dict:
    image = Image.open(image_path).convert("RGB")
    georef = GeoReference(west=0, south=0, east=1, north=1)
    mask_result = build_urban_mask(image, max_width=max_width)
    contours = detect_urban_contours(image, georef, max_width=max_width, min_area_pixels=42, max_contours=24)

    base = mask_result.image
    mask_img = Image.new("RGBA", base.size, (0, 0, 0, 255))
    mask_pixels = np.asarray(mask_result.mask, dtype=np.uint8) * 255
    score_pixels = (np.clip(mask_result.score, 0, 1) * 255).astype(np.uint8)
    mask_img = Image.merge(
        "RGBA",
        (
            Image.fromarray(score_pixels, mode="L"),
            Image.fromarray(mask_pixels, mode="L"),
            Image.fromarray(np.zeros_like(score_pixels), mode="L"),
            Image.fromarray(np.full_like(score_pixels, 255), mode="L"),
        ),
    )

    overlay = base.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    for contour in contours:
        points = [_point_to_pixel(x, y, base.width, base.height, georef) for x, y in contour.polygon_mercator]
        if len(points) >= 3:
            draw.polygon(points, fill=(16, 185, 129, 58), outline=(250, 204, 21, 235))
            draw.line(points + [points[0]], fill=(250, 204, 21, 255), width=3)
        min_x, min_y, max_x, max_y = contour.bbox_mercator
        top_left = _point_to_pixel(min_x, max_y, base.width, base.height, georef)
        bottom_right = _point_to_pixel(max_x, min_y, base.width, base.height, georef)
        draw.rectangle([top_left, bottom_right], outline=(34, 211, 238, 210), width=1)

    panel = Image.new("RGB", (base.width * 3, base.height), (18, 20, 24))
    panel.paste(base, (0, 0))
    panel.paste(mask_img.convert("RGB"), (base.width, 0))
    panel.paste(overlay.convert("RGB"), (base.width * 2, 0))
    output_path = output_dir / f"{image_path.stem}-urban-poc.jpg"
    panel.save(output_path, quality=92)

    return {
        "input": str(image_path),
        "output": str(output_path),
        "size": [base.width, base.height],
        "threshold": round(mask_result.threshold, 4),
        "maskRatio": round(float(mask_result.mask.mean()), 4),
        "contours": [
            {
                "id": contour.contour_id,
                "areaPixels": contour.area_pixels,
                "score": round(contour.urban_score, 4),
                "points": len(contour.polygon_mercator),
            }
            for contour in contours[:8]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output/urban-vision-poc"))
    parser.add_argument("--max-width", type=int, default=420)
    args = parser.parse_args()

    images = args.images or [
        Path("data/raw/real-v1/wms-mosaics/zurich-center.jpg"),
        Path("data/raw/real-v1/wms-mosaics/winterthur-center.jpg"),
        Path("data/raw/real-v1/wms-mosaics/chur-center.jpg"),
        Path("data/cache/autopilot/swissimage-z9-w1536.jpg"),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = [_render_debug(path, args.output_dir, args.max_width) for path in images]
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"outputDir": str(args.output_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

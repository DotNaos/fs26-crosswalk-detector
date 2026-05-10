from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from crosswalk_detector.autopilot import EARTH_RADIUS_M, GeoReference
from crosswalk_detector.urban_vision import detect_urban_contours, detect_urban_envelopes


def _mercator_to_lon_lat(x: float, y: float) -> tuple[float, float]:
    lon = math.degrees(x / EARTH_RADIUS_M)
    lat = math.degrees(2 * math.atan(math.exp(y / EARTH_RADIUS_M)) - math.pi / 2)
    return lon, lat


def _point_to_pixel(x: float, y: float, width: int, height: int, georef: GeoReference) -> tuple[int, int]:
    lon, lat = _mercator_to_lon_lat(x, y)
    return (
        round((lon - georef.west) / (georef.east - georef.west) * width),
        round((georef.north - lat) / (georef.north - georef.south) * height),
    )


def _draw_contours(base: Image.Image, contours, georef: GeoReference, title: str, fill=(16, 185, 129, 48)) -> Image.Image:
    overlay = base.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    for contour in contours:
        points = [_point_to_pixel(x, y, base.width, base.height, georef) for x, y in contour.polygon_mercator]
        if len(points) < 3:
            continue
        draw.polygon(points, fill=fill, outline=(250, 204, 21, 240))
        draw.line(points + [points[0]], fill=(250, 204, 21, 255), width=3)
    draw.rectangle((0, 0, base.width, 24), fill=(0, 0, 0, 165))
    draw.text((8, 6), title, fill=(255, 255, 255, 255))
    return overlay.convert("RGB")


def _fit(image: Image.Image, width: int) -> Image.Image:
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.BILINEAR)


def _render_image_variants(image_path: Path, output_dir: Path, display_width: int, widths: list[int]) -> dict:
    image = Image.open(image_path).convert("RGB")
    georef = GeoReference(west=0, south=0, east=1, north=1)
    base = _fit(image, display_width)
    variants = [base]
    details = []

    detailed = detect_urban_contours(image, georef, max_width=max(widths), min_area_pixels=42, max_contours=16)
    variants.append(_draw_contours(base, detailed, georef, f"detail {max(widths)}px", fill=(6, 182, 212, 42)))
    details.append({"kind": "detail", "width": max(widths), "contours": len(detailed), "largest": detailed[0].area_pixels if detailed else 0})

    for width in widths:
        envelopes = detect_urban_envelopes(image, georef, max_width=width, max_contours=10)
        variants.append(_draw_contours(base, envelopes, georef, f"city envelope {width}px"))
        details.append({"kind": "envelope", "width": width, "contours": len(envelopes), "largest": envelopes[0].area_pixels if envelopes else 0})

    gutter = 8
    panel = Image.new("RGB", (sum(img.width for img in variants) + gutter * (len(variants) - 1), max(img.height for img in variants)), (18, 20, 24))
    x = 0
    for img in variants:
        panel.paste(img, (x, 0))
        x += img.width + gutter
    output_path = output_dir / f"{image_path.stem}-scale-poc.jpg"
    panel.save(output_path, quality=92)
    return {"input": str(image_path), "output": str(output_path), "variants": details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output/urban-scale-poc"))
    parser.add_argument("--display-width", type=int, default=300)
    parser.add_argument("--widths", default="96,144,220,320")
    args = parser.parse_args()

    images = args.images or [
        Path("data/raw/real-v1/wms-mosaics/zurich-center.jpg"),
        Path("data/raw/real-v1/wms-mosaics/winterthur-center.jpg"),
        Path("data/raw/real-v1/wms-mosaics/chur-center.jpg"),
        Path("data/raw/real-v1/wms-mosaics/basel-center.jpg"),
        Path("data/cache/autopilot/swissimage-z9-w1536.jpg"),
    ]
    widths = [int(part) for part in args.widths.split(",") if part.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = [_render_image_variants(image, args.output_dir, args.display_width, widths) for image in images]
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"outputDir": str(args.output_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

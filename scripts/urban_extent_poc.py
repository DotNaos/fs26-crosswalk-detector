from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tomllib
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from crosswalk_detector.autopilot import EARTH_RADIUS_M, GeoReference
from crosswalk_detector.urban_vision import build_urban_road_line_mask, detect_urban_envelopes


WMS_BASE_URL = "https://wms.geo.admin.ch/"


def _mercator_from_lat_lon(latitude: float, longitude: float) -> tuple[float, float]:
    return (
        EARTH_RADIUS_M * math.radians(longitude),
        EARTH_RADIUS_M * math.log(math.tan(math.pi / 4.0 + math.radians(latitude) / 2.0)),
    )


def _mercator_to_lon_lat(x: float, y: float) -> tuple[float, float]:
    return (
        math.degrees(x / EARTH_RADIUS_M),
        math.degrees(2 * math.atan(math.exp(y / EARTH_RADIUS_M)) - math.pi / 2),
    )


def _build_wms_url(latitude: float, longitude: float, size_m: int, image_px: int) -> str:
    center_x, center_y = _mercator_from_lat_lon(latitude, longitude)
    half = size_m / 2
    bbox = (center_x - half, center_y - half, center_x + half, center_y + half)
    query = urlencode(
        {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": "ch.swisstopo.swissimage-product",
            "STYLES": "default",
            "CRS": "EPSG:3857",
            "BBOX": ",".join(str(value) for value in bbox),
            "WIDTH": image_px,
            "HEIGHT": image_px,
            "FORMAT": "image/jpeg",
        }
    )
    return f"{WMS_BASE_URL}?{query}"


def _fetch_extent(scene: dict, size_m: int, image_px: int, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{scene['scene_id']}-{size_m}m-{image_px}px.jpg"
    if path.exists():
        return path
    with urlopen(_build_wms_url(scene["latitude"], scene["longitude"], size_m, image_px), timeout=18) as response:
        path.write_bytes(response.read())
    return path


def _point_to_pixel(x: float, y: float, width: int, height: int, georef: GeoReference) -> tuple[int, int]:
    lon, lat = _mercator_to_lon_lat(x, y)
    return (
        round((lon - georef.west) / (georef.east - georef.west) * width),
        round((georef.north - lat) / (georef.north - georef.south) * height),
    )


def _dummy_georef() -> GeoReference:
    return GeoReference(west=0, south=0, east=1, north=1)


def _draw_envelopes(image: Image.Image, contours, title: str) -> Image.Image:
    base = image.convert("RGBA")
    draw = ImageDraw.Draw(base, "RGBA")
    georef = _dummy_georef()
    for contour in contours:
        points = [_point_to_pixel(x, y, base.width, base.height, georef) for x, y in contour.polygon_mercator]
        if len(points) < 3:
            continue
        draw.line(points + [points[0]], fill=(8, 12, 18, 245), width=5)
        draw.line(points + [points[0]], fill=(250, 204, 21, 255), width=3)
    draw.rectangle((0, 0, base.width, 26), fill=(0, 0, 0, 170))
    draw.text((8, 7), title, fill=(255, 255, 255, 255))
    return base.convert("RGB")


def _thin_mask(mask: np.ndarray, max_iterations: int = 80) -> np.ndarray:
    current = mask.astype(bool).copy()
    for _ in range(max_iterations):
        changed = False
        for step in (0, 1):
            p2 = np.roll(current, -1, axis=0)
            p3 = np.roll(np.roll(current, -1, axis=0), 1, axis=1)
            p4 = np.roll(current, 1, axis=1)
            p5 = np.roll(np.roll(current, 1, axis=0), 1, axis=1)
            p6 = np.roll(current, 1, axis=0)
            p7 = np.roll(np.roll(current, 1, axis=0), -1, axis=1)
            p8 = np.roll(current, -1, axis=1)
            p9 = np.roll(np.roll(current, -1, axis=0), -1, axis=1)
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            count = sum(neighbors)
            transitions = sum((~neighbors[index] & neighbors[(index + 1) % 8]) for index in range(8))
            if step == 0:
                keep_connectivity = ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                keep_connectivity = ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            remove = current & (count >= 2) & (count <= 6) & (transitions == 1) & keep_connectivity
            remove[[0, -1], :] = False
            remove[:, [0, -1]] = False
            if remove.any():
                current[remove] = False
                changed = True
        if not changed:
            break
    return current


def _draw_road_veins(image: Image.Image, road_mask, title: str) -> Image.Image:
    base = image.convert("RGBA")
    thinned = _thin_mask(road_mask.mask)
    mask = Image.fromarray(thinned.astype(np.uint8) * 255)
    mask = mask.resize(base.size, Image.Resampling.NEAREST)

    glow = mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.8))
    glow_layer = Image.new("RGBA", base.size, (56, 189, 248, 0))
    glow_layer.putalpha(glow.point(lambda value: min(120, round(value * 0.35))))

    vein_layer = Image.new("RGBA", base.size, (14, 165, 233, 0))
    vein_layer.putalpha(mask.point(lambda value: min(235, round(value * 0.9))))

    base = Image.alpha_composite(base, glow_layer)
    base = Image.alpha_composite(base, vein_layer)
    draw = ImageDraw.Draw(base, "RGBA")
    draw.rectangle((0, 0, base.width, 26), fill=(0, 0, 0, 170))
    draw.text((8, 7), title, fill=(255, 255, 255, 255))
    return base.convert("RGB")


def _draw_road_score(image: Image.Image, road_mask, title: str) -> Image.Image:
    base = image.convert("RGBA")
    score = road_mask.score
    low = float(np.quantile(score, 0.88))
    high = float(np.quantile(score, 0.995))
    normalized = np.clip((score - low) / max(1e-6, high - low), 0, 1)
    alpha = Image.fromarray((normalized * 210).astype(np.uint8)).resize(base.size, Image.Resampling.BILINEAR)

    heat = Image.new("RGBA", base.size, (14, 165, 233, 0))
    heat.putalpha(alpha)
    base = Image.alpha_composite(base, heat)
    draw = ImageDraw.Draw(base, "RGBA")
    draw.rectangle((0, 0, base.width, 26), fill=(0, 0, 0, 170))
    draw.text((8, 7), title, fill=(255, 255, 255, 255))
    return base.convert("RGB")


def _fit(image: Image.Image, width: int) -> Image.Image:
    return image.resize((width, round(image.height * width / image.width)), Image.Resampling.BILINEAR)


def _scene_montage(scene: dict, sizes: list[int], image_px: int, display_width: int, cache_dir: Path, output_dir: Path) -> dict:
    panels = []
    stats = []
    for size_m in sizes:
        try:
            image_path = _fetch_extent(scene, size_m, image_px, cache_dir)
        except TimeoutError:
            stats.append({"sizeM": size_m, "error": "WMS timeout"})
            continue
        image = Image.open(image_path).convert("RGB")
        texture_contours = detect_urban_envelopes(image, _dummy_georef(), max_width=180, min_area_ratio=0.018, max_contours=8, method="texture")
        palette_contours = detect_urban_envelopes(image, _dummy_georef(), max_width=180, min_area_ratio=0.018, max_contours=8, method="palette_pattern")
        road_seeded_contours = detect_urban_envelopes(image, _dummy_georef(), max_width=320, min_area_ratio=0.006, max_contours=8, method="road_seeded")
        road_density_contours = detect_urban_envelopes(image, _dummy_georef(), max_width=320, min_area_ratio=0.006, max_contours=8, method="road_density")
        road_vein_contours = detect_urban_envelopes(image, _dummy_georef(), max_width=320, min_area_ratio=0.006, max_contours=8, method="road_veins")
        road_mask = build_urban_road_line_mask(image, max_width=420, threshold=0.52)
        fitted = _fit(image, display_width)
        panels.append(_draw_envelopes(fitted, texture_contours, f"{scene['city']} · {size_m // 1000:g} km · texture"))
        panels.append(_draw_envelopes(fitted, palette_contours, f"{scene['city']} · {size_m // 1000:g} km · palette+pattern"))
        panels.append(_draw_envelopes(fitted, road_seeded_contours, f"{scene['city']} · {size_m // 1000:g} km · road seeded"))
        panels.append(_draw_envelopes(fitted, road_density_contours, f"{scene['city']} · {size_m // 1000:g} km · road density"))
        panels.append(_draw_envelopes(fitted, road_vein_contours, f"{scene['city']} · {size_m // 1000:g} km · road veins"))
        panels.append(_draw_road_veins(fitted, road_mask, f"{scene['city']} · {size_m // 1000:g} km · vein mask"))
        panels.append(_draw_road_score(fitted, road_mask, f"{scene['city']} · {size_m // 1000:g} km · vein score"))
        stats.append(
            {
                "sizeM": size_m,
                "image": str(image_path),
                "textureContours": len(texture_contours),
                "textureLargestAreaPixels": texture_contours[0].area_pixels if texture_contours else 0,
                "palettePatternContours": len(palette_contours),
                "palettePatternLargestAreaPixels": palette_contours[0].area_pixels if palette_contours else 0,
                "roadSeededContours": len(road_seeded_contours),
                "roadSeededLargestAreaPixels": road_seeded_contours[0].area_pixels if road_seeded_contours else 0,
                "roadDensityContours": len(road_density_contours),
                "roadDensityLargestAreaPixels": road_density_contours[0].area_pixels if road_density_contours else 0,
                "roadVeinContours": len(road_vein_contours),
                "roadVeinLargestAreaPixels": road_vein_contours[0].area_pixels if road_vein_contours else 0,
                "roadMaskCoverage": float(road_mask.mask.mean()),
            }
        )

    gutter = 8
    if not panels:
        return {"scene": scene["scene_id"], "output": None, "stats": stats}
    montage = Image.new("RGB", (sum(panel.width for panel in panels) + gutter * (len(panels) - 1), max(panel.height for panel in panels)), (18, 20, 24))
    x = 0
    for panel in panels:
        montage.paste(panel, (x, 0))
        x += panel.width + gutter
    output_path = output_dir / f"{scene['scene_id']}-extent-poc.jpg"
    montage.save(output_path, quality=92)
    return {"scene": scene["scene_id"], "output": str(output_path), "stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/real-dataset.toml"))
    parser.add_argument("--cities", default="zurich-center,winterthur-center,chur-center,basel-center")
    parser.add_argument("--sizes", default="3000,8000,18000,40000")
    parser.add_argument("--image-px", type=int, default=768)
    parser.add_argument("--display-width", type=int, default=300)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/urban-extent-poc"))
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output/urban-extent-poc"))
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    wanted = set(args.cities.split(","))
    scenes = [scene for scene in config["scenes"] if scene["scene_id"] in wanted]
    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = [_scene_montage(scene, sizes, args.image_px, args.display_width, args.cache_dir, args.output_dir) for scene in scenes]
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"outputDir": str(args.output_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

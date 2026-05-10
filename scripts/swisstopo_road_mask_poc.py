from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tomllib
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, UnidentifiedImageError

from crosswalk_detector.autopilot import EARTH_RADIUS_M


WMS_BASE_URL = "https://wms.geo.admin.ch/"
SWISSTLM3D_ROADS_LAYER = "ch.swisstopo.swisstlm3d-strassen"


def _mercator_from_lat_lon(latitude: float, longitude: float) -> tuple[float, float]:
    return (
        EARTH_RADIUS_M * math.radians(longitude),
        EARTH_RADIUS_M * math.log(math.tan(math.pi / 4.0 + math.radians(latitude) / 2.0)),
    )


def _build_wms_url(latitude: float, longitude: float, size_m: int, image_px: int, layer: str, image_format: str) -> str:
    center_x, center_y = _mercator_from_lat_lon(latitude, longitude)
    half = size_m / 2
    bbox = (center_x - half, center_y - half, center_x + half, center_y + half)
    query = urlencode(
        {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": layer,
            "STYLES": "default",
            "CRS": "EPSG:3857",
            "BBOX": ",".join(str(value) for value in bbox),
            "WIDTH": image_px,
            "HEIGHT": image_px,
            "FORMAT": image_format,
            "TRANSPARENT": "TRUE",
            "LANG": "en",
        }
    )
    return f"{WMS_BASE_URL}?{query}"


def _fetch_wms(scene: dict, size_m: int, image_px: int, layer: str, image_format: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "png" if image_format == "image/png" else "jpg"
    path = cache_dir / f"{scene['scene_id']}-{layer}-{size_m}m-{image_px}px.{suffix}"
    if path.exists():
        return path
    url = _build_wms_url(scene["latitude"], scene["longitude"], size_m, image_px, layer, image_format)
    with urlopen(url, timeout=20) as response:
        path.write_bytes(response.read())
    return path


def _official_road_mask(roads_image: Image.Image) -> np.ndarray:
    rgba = roads_image.convert("RGBA")
    pixels = np.asarray(rgba, dtype=np.uint8)
    alpha = pixels[:, :, 3] > 8
    rgb = pixels[:, :, :3].astype(np.int16)
    dark_or_colored = (rgb.max(axis=2) - rgb.min(axis=2) > 12) | (rgb.mean(axis=2) < 245)
    return alpha & dark_or_colored


def _draw_overlay(satellite: Image.Image, official_roads: Image.Image, title: str) -> Image.Image:
    base = satellite.convert("RGBA")
    mask = Image.fromarray(_official_road_mask(official_roads).astype(np.uint8) * 255)
    mask = mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.45))
    road_layer = Image.new("RGBA", base.size, (14, 165, 233, 0))
    road_layer.putalpha(mask.point(lambda value: min(245, round(value * 0.92))))
    base = Image.alpha_composite(base, road_layer)
    draw = ImageDraw.Draw(base, "RGBA")
    draw.rectangle((0, 0, base.width, 28), fill=(0, 0, 0, 175))
    draw.text((8, 8), title, fill=(255, 255, 255, 255))
    return base.convert("RGB")


def _open_satellite(path: Path, scene: dict, size_m: int, image_px: int, fallback_dir: Path) -> tuple[Image.Image, str]:
    try:
        return Image.open(path).convert("RGB"), str(path)
    except UnidentifiedImageError:
        fallback = fallback_dir / f"{scene['scene_id']}-{size_m}m-{image_px}px.jpg"
        if fallback.exists():
            return Image.open(fallback).convert("RGB"), str(fallback)
        fallback_candidates = sorted(fallback_dir.glob(f"{scene['scene_id']}-{size_m}m-*.jpg"))
        if fallback_candidates:
            image = Image.open(fallback_candidates[0]).convert("RGB")
            return image.resize((image_px, image_px), Image.Resampling.BILINEAR), str(fallback_candidates[0])
        raise


def _scene_montage(scene: dict, size_m: int, image_px: int, cache_dir: Path, output_dir: Path, satellite_fallback_dir: Path) -> dict:
    satellite_path = _fetch_wms(scene, size_m, image_px, "ch.swisstopo.swissimage-product", "image/jpeg", cache_dir)
    roads_path = _fetch_wms(scene, size_m, image_px, SWISSTLM3D_ROADS_LAYER, "image/png", cache_dir)
    satellite, satellite_source = _open_satellite(satellite_path, scene, size_m, image_px, satellite_fallback_dir)
    roads = Image.open(roads_path).convert("RGBA")

    original = satellite.copy()
    draw = ImageDraw.Draw(original)
    draw.rectangle((0, 0, original.width, 28), fill=(0, 0, 0))
    draw.text((8, 8), f"{scene['city']} · satellite", fill=(255, 255, 255))

    official = _draw_overlay(satellite, roads, f"{scene['city']} · official swissTLM3D roads")
    raw_roads = Image.new("RGB", roads.size, (18, 20, 24))
    raw_roads.paste(roads.convert("RGB"), mask=roads.getchannel("A"))
    draw = ImageDraw.Draw(raw_roads)
    draw.rectangle((0, 0, raw_roads.width, 28), fill=(0, 0, 0))
    draw.text((8, 8), f"{scene['city']} · raw road WMS", fill=(255, 255, 255))

    gutter = 10
    montage = Image.new("RGB", (image_px * 3 + gutter * 2, image_px), (18, 20, 24))
    for index, panel in enumerate((original, official, raw_roads)):
        montage.paste(panel, (index * (image_px + gutter), 0))
    output_path = output_dir / f"{scene['scene_id']}-official-road-mask.jpg"
    montage.save(output_path, quality=94)
    return {
        "scene": scene["scene_id"],
        "satellite": satellite_source,
        "roads": str(roads_path),
        "output": str(output_path),
        "officialRoadCoverage": float(_official_road_mask(roads).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/real-dataset.toml"))
    parser.add_argument("--cities", default="zurich-center,winterthur-center,chur-center")
    parser.add_argument("--size-m", type=int, default=8000)
    parser.add_argument("--image-px", type=int, default=768)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/swisstopo-road-mask-poc"))
    parser.add_argument("--satellite-fallback-dir", type=Path, default=Path("data/cache/urban-extent-poc-small"))
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output/swisstopo-road-mask-poc"))
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    wanted = set(args.cities.split(","))
    scenes = [scene for scene in config["scenes"] if scene["scene_id"] in wanted]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for scene in scenes:
        try:
            summary.append(_scene_montage(scene, args.size_m, args.image_px, args.cache_dir, args.output_dir, args.satellite_fallback_dir))
        except URLError as error:
            summary.append(
                {
                    "scene": scene["scene_id"],
                    "error": f"WMS fetch failed: {error}",
                    "hint": "Run again when network/VPN access to wms.geo.admin.ch is available, or keep the cached WMS files in the cache directory.",
                }
            )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"outputDir": str(args.output_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

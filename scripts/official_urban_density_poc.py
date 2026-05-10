from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from crosswalk_detector.urban_vision import (
    _box_blur,
    _close,
    _component_masks,
    _dilate,
    _erode,
    _open,
    build_urban_palette_pattern_mask,
    build_urban_road_density_mask,
)


def _official_road_mask(roads_image: Image.Image) -> np.ndarray:
    rgba = roads_image.convert("RGBA")
    pixels = np.asarray(rgba, dtype=np.uint8)
    alpha = pixels[:, :, 3] > 8
    rgb = pixels[:, :, :3].astype(np.int16)
    dark_or_colored = (rgb.max(axis=2) - rgb.min(axis=2) > 12) | (rgb.mean(axis=2) < 245)
    return alpha & dark_or_colored


def _official_main_road_mask(roads_image: Image.Image) -> np.ndarray:
    rgba = roads_image.convert("RGBA")
    pixels = np.asarray(rgba, dtype=np.uint8)
    alpha = pixels[:, :, 3] > 8
    red = pixels[:, :, 0].astype(np.int16)
    green = pixels[:, :, 1].astype(np.int16)
    blue = pixels[:, :, 2].astype(np.int16)
    mean = (red + green + blue) / 3
    chroma = np.maximum.reduce((red, green, blue)) - np.minimum.reduce((red, green, blue))

    yellow_or_orange = alpha & (red > 175) & (green > 105) & (blue < 120)
    pink = alpha & (red > 170) & (green > 95) & (green < 185) & (blue > 110) & (blue < 190)
    bright_local = alpha & (mean > 165) & (chroma < 55)
    return yellow_or_orange | pink | bright_local


def _official_reliable_road_masks(roads_image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgba = roads_image.convert("RGBA")
    pixels = np.asarray(rgba, dtype=np.uint8)
    alpha = pixels[:, :, 3] > 8
    red = pixels[:, :, 0].astype(np.int16)
    green = pixels[:, :, 1].astype(np.int16)
    blue = pixels[:, :, 2].astype(np.int16)
    mean = (red + green + blue) / 3
    chroma = np.maximum.reduce((red, green, blue)) - np.minimum.reduce((red, green, blue))

    bright_local_roads = alpha & (mean > 150) & (chroma < 80)
    colored_major_roads = alpha & (chroma >= 80)
    return bright_local_roads | colored_major_roads, bright_local_roads, colored_major_roads


def _urban_density_mask(roads_image: Image.Image, *, surface_quantile: float = 0.44, minimum_threshold: float = 0.16) -> tuple[np.ndarray, np.ndarray, float]:
    reliable_roads, _bright_local_roads, colored_major_roads = _official_reliable_road_masks(roads_image)
    width = roads_image.width
    local_density = _box_blur(reliable_roads.astype(np.float32), radius=max(5, round(width / 52)))
    mid_density = _box_blur(reliable_roads.astype(np.float32), radius=max(8, round(width / 34)))
    broad_density = _box_blur(reliable_roads.astype(np.float32), radius=max(12, round(width / 22)))
    major_context = _box_blur(colored_major_roads.astype(np.float32), radius=max(8, round(width / 18)))
    density = np.clip(local_density * 0.72 + mid_density * 0.72 + broad_density * 0.34 + major_context * 0.28, 0, 1)

    nonzero = density[density > 0]
    threshold = max(minimum_threshold, float(np.quantile(nonzero, surface_quantile)) if nonzero.size else 1.0)
    seed_threshold = max(threshold, float(np.quantile(nonzero, 0.72)) if nonzero.size else threshold)
    mask = density >= threshold
    seed = density >= seed_threshold
    mask = _open(_close(mask, iterations=max(1, round(width / 180))), iterations=1)

    kept = np.zeros(mask.shape, dtype=bool)
    seed_context = _dilate(seed, iterations=max(2, round(width / 80)))
    min_area = max(96, round(mask.size * 0.003))
    for component in _component_masks(mask, min_area=min_area):
        if (component & seed_context).any():
            kept |= component
    return kept, density, threshold


def _draw_title(image: Image.Image, title: str) -> Image.Image:
    base = image.convert("RGBA")
    draw = ImageDraw.Draw(base, "RGBA")
    draw.rectangle((0, 0, base.width, 28), fill=(0, 0, 0, 175))
    draw.text((8, 8), title, fill=(255, 255, 255, 255))
    return base.convert("RGB")


def _draw_road_overlay(satellite: Image.Image, roads: Image.Image, title: str) -> Image.Image:
    base = satellite.convert("RGBA")
    mask = Image.fromarray(_official_road_mask(roads).astype(np.uint8) * 255)
    mask = mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.45))
    layer = Image.new("RGBA", base.size, (14, 165, 233, 0))
    layer.putalpha(mask.point(lambda value: min(210, round(value * 0.76))))
    base = Image.alpha_composite(base, layer)
    return _draw_title(base.convert("RGB"), title)


def _draw_density_envelopes(satellite: Image.Image, roads: Image.Image, title: str) -> Image.Image:
    mask, density, threshold = _urban_density_mask(roads)
    return _draw_mask_envelopes(satellite, mask, density, threshold, title)


def _fused_urban_mask(satellite: Image.Image, roads: Image.Image) -> tuple[np.ndarray, np.ndarray, float]:
    official_mask, official_density, threshold = _urban_density_mask(roads)
    road_density = build_urban_road_density_mask(satellite, max_width=satellite.width, threshold=0.52)
    palette = build_urban_palette_pattern_mask(satellite, max_width=satellite.width, threshold=0.52)
    satellite_context = road_density.mask | palette.mask
    satellite_context = _dilate(satellite_context, iterations=max(6, round(satellite.width / 42)))
    fused = official_mask & satellite_context
    fused = _open(_close(fused, iterations=max(2, round(satellite.width / 130))), iterations=1)

    kept = np.zeros(fused.shape, dtype=bool)
    min_area = max(64, round(fused.size * 0.004))
    for component in _component_masks(fused, min_area=min_area):
        kept |= component
    fused_score = np.clip(official_density * 0.72 + road_density.score * 0.46 + palette.score * 0.25, 0, 1)
    return kept, fused_score, threshold


def _draw_fused_envelopes(satellite: Image.Image, roads: Image.Image, title: str) -> Image.Image:
    mask, score, threshold = _fused_urban_mask(satellite, roads)
    return _draw_mask_envelopes(satellite, mask, score, threshold, title)


def _draw_mask_envelopes(satellite: Image.Image, mask: np.ndarray, density: np.ndarray, threshold: float, title: str) -> Image.Image:
    base = satellite.convert("RGBA")
    alpha = Image.fromarray((np.clip((density - threshold) / max(1e-6, 1 - threshold), 0, 1) * 90).astype(np.uint8))
    fill = Image.new("RGBA", base.size, (34, 197, 94, 0))
    fill.putalpha(alpha)
    base = Image.alpha_composite(base, fill)

    boundary = mask & ~_erode(mask, iterations=1)
    shadow_mask = Image.fromarray(_dilate(boundary, iterations=2).astype(np.uint8) * 255)
    shadow_layer = Image.new("RGBA", base.size, (8, 12, 18, 0))
    shadow_layer.putalpha(shadow_mask.point(lambda value: min(240, value)))
    base = Image.alpha_composite(base, shadow_layer)

    line_mask = Image.fromarray(_dilate(boundary, iterations=1).astype(np.uint8) * 255)
    line_layer = Image.new("RGBA", base.size, (34, 197, 94, 0))
    line_layer.putalpha(line_mask.point(lambda value: min(255, value)))
    base = Image.alpha_composite(base, line_layer)

    return _draw_title(base.convert("RGB"), title)


def _raw_roads_panel(roads: Image.Image, title: str) -> Image.Image:
    raw = Image.new("RGB", roads.size, (18, 20, 24))
    raw.paste(roads.convert("RGB"), mask=roads.convert("RGBA").getchannel("A"))
    return _draw_title(raw, title)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--satellite-dir", type=Path, default=Path("data/cache/urban-extent-poc-small"))
    parser.add_argument("--road-dir", type=Path, default=Path("data/cache/swisstopo-road-mask-poc"))
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output/official-urban-density-poc"))
    parser.add_argument("--cities", default="zurich-center,winterthur-center,chur-center")
    parser.add_argument("--size-m", type=int, default=8000)
    parser.add_argument("--image-px", type=int, default=512)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for scene_id in [city for city in args.cities.split(",") if city]:
        satellite = Image.open(args.satellite_dir / f"{scene_id}-{args.size_m}m-{args.image_px}px.jpg").convert("RGB")
        roads = Image.open(args.road_dir / f"{scene_id}-ch.swisstopo.swisstlm3d-strassen-{args.size_m}m-{args.image_px}px.png").convert("RGBA")
        mask, _density, threshold = _urban_density_mask(roads)
        panels = [
            _draw_title(satellite, f"{scene_id} · satellite"),
            _raw_roads_panel(roads, f"{scene_id} · official roads"),
            _draw_road_overlay(satellite, roads, f"{scene_id} · all official roads"),
            _draw_density_envelopes(satellite, roads, f"{scene_id} · dense urban road envelope"),
            _draw_fused_envelopes(satellite, roads, f"{scene_id} · fused satellite+official envelope"),
        ]
        gutter = 10
        montage = Image.new("RGB", (args.image_px * len(panels) + gutter * (len(panels) - 1), args.image_px), (18, 20, 24))
        x = 0
        for panel in panels:
            montage.paste(panel, (x, 0))
            x += panel.width + gutter
        output = args.output_dir / f"{scene_id}-official-urban-density.jpg"
        montage.save(output, quality=94)
        fused_mask, _fused_score, _fused_threshold = _fused_urban_mask(satellite, roads)
        summaries.append(
            {
                "scene": scene_id,
                "threshold": threshold,
                "officialDensityCoverage": float(mask.mean()),
                "fusedCoverage": float(fused_mask.mean()),
                "output": str(output),
            }
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps({"outputDir": str(args.output_dir), "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()

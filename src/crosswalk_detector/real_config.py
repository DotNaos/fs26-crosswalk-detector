"""Config loading for the real crosswalk dataset pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    city: str
    split: str
    latitude: float
    longitude: float
    size_m: int = 800
    image_px: int = 2048


@dataclass(frozen=True)
class SelectionConfig:
    positive_min_combined: float = 0.6
    positive_min_road_surface: float = -0.02
    positive_min_heuristic: float = 0.5
    negative_max_combined: float = 0.55
    negative_positive_penalty: float = 0.35


@dataclass(frozen=True)
class RealPipelineConfig:
    run_name: str
    export_name: str
    target_per_class: int
    tile_size_m: int
    split_ratios: dict[str, float]
    selection: SelectionConfig
    scenes: tuple[SceneSpec, ...]
    path: Path


DEFAULT_CONFIG_PATH = Path("configs/real-dataset.toml")


def load_real_pipeline_config(path: Path | None = None) -> RealPipelineConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    scenes = tuple(
        SceneSpec(
            scene_id=str(scene["scene_id"]),
            city=str(scene["city"]),
            split=str(scene["split"]),
            latitude=float(scene["latitude"]),
            longitude=float(scene["longitude"]),
            size_m=int(scene.get("size_m", 800)),
            image_px=int(scene.get("image_px", 2048)),
        )
        for scene in raw.get("scenes", [])
    )
    if not scenes:
        raise ValueError(f"No scenes configured in {config_path}")

    split_ratios_raw = raw.get("split_ratios", {})
    selection_raw = raw.get("selection", {})

    return RealPipelineConfig(
        run_name=str(raw.get("run_name", "real-v1")),
        export_name=str(raw.get("export_name", "real-balanced-256")),
        target_per_class=int(raw.get("target_per_class", 128)),
        tile_size_m=int(raw.get("tile_size_m", 25)),
        split_ratios={key: float(value) for key, value in split_ratios_raw.items()},
        selection=SelectionConfig(
            positive_min_combined=float(selection_raw.get("positive_min_combined", 0.6)),
            positive_min_road_surface=float(selection_raw.get("positive_min_road_surface", -0.02)),
            positive_min_heuristic=float(selection_raw.get("positive_min_heuristic", 0.5)),
            negative_max_combined=float(selection_raw.get("negative_max_combined", 0.55)),
            negative_positive_penalty=float(selection_raw.get("negative_positive_penalty", 0.35)),
        ),
        scenes=scenes,
        path=config_path,
    )

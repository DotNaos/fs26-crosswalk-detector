"""Local-first pilot planning helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen

RAW_MB_PER_KM2_AT_10CM = 55.0
TILE_SIZE_M = 25
AREA_SIDE_M = 1_000
WMTS_TEMPLATE = (
    "https://wmts.geo.admin.ch/1.0.0/"
    "ch.swisstopo.swissimage-product/default/current/3857/"
    "{zoom}/{tile_col}/{tile_row}.jpeg"
)


@dataclass(frozen=True)
class CityPlan:
    name: str
    country: str
    latitude: float
    longitude: float
    area_km2: float
    split: str
    notes: str


@dataclass(frozen=True)
class PilotConfig:
    run_name: str
    max_raw_size_gb: float
    imagery_resolution_cm: int
    tile_size_m: int
    sliding_window_step_m: int
    target_per_class: int
    split_ratios: dict[str, float]
    wmts_zoom: int
    wmts_tile_radius: int
    auto_search_radius: int
    cities: list[CityPlan]


@dataclass(frozen=True)
class CrosswalkCentroidModel:
    feature_names: tuple[str, ...]
    positive_mean: tuple[float, ...]
    negative_mean: tuple[float, ...]
    scale: tuple[float, ...]
    training_samples: int


def image_label_pair_count(items: list[tuple[str, int]]) -> int:
    """Return the number of labeled image items."""

    return len(items)


def load_pilot_config(path: Path) -> PilotConfig:
    """Load a pilot config from JSON."""

    data = json.loads(path.read_text())
    cities = [CityPlan(**city) for city in data["cities"]]
    return PilotConfig(
        run_name=data["run_name"],
        max_raw_size_gb=float(data["max_raw_size_gb"]),
        imagery_resolution_cm=int(data["imagery_resolution_cm"]),
        tile_size_m=int(data["tile_size_m"]),
        sliding_window_step_m=int(data["sliding_window_step_m"]),
        target_per_class=int(data["target_per_class"]),
        split_ratios={str(key): float(value) for key, value in data["split_ratios"].items()},
        wmts_zoom=int(data["wmts_zoom"]),
        wmts_tile_radius=int(data["wmts_tile_radius"]),
        auto_search_radius=int(data["auto_search_radius"]),
        cities=cities,
    )


def derive_split_targets(target_per_class: int, split_ratios: dict[str, float]) -> dict[str, int]:
    """Derive integer split targets that sum to the per-class target."""

    raw_targets = {
        split: target_per_class * ratio for split, ratio in split_ratios.items()
    }
    split_targets = {
        split: int(math.floor(value)) for split, value in raw_targets.items()
    }
    assigned = sum(split_targets.values())
    remainder = target_per_class - assigned
    split_order = sorted(
        split_ratios,
        key=lambda split: (raw_targets[split] - split_targets[split], split),
        reverse=True,
    )
    for split in split_order[:remainder]:
        split_targets[split] += 1
    return split_targets


def raw_size_mb(area_km2: float) -> float:
    """Estimate raw imagery size in megabytes at 10 cm resolution."""

    return area_km2 * RAW_MB_PER_KM2_AT_10CM


def non_overlapping_tile_count(area_km2: float, tile_size_m: int) -> int:
    """Estimate the number of non-overlapping task tiles."""

    tiles_per_side = AREA_SIDE_M // tile_size_m
    return int(area_km2 * (tiles_per_side**2))


def sliding_window_count(area_km2: float, tile_size_m: int, step_m: int) -> int:
    """Estimate the number of sliding windows for a given step size."""

    windows_per_side = ((AREA_SIDE_M - tile_size_m) // step_m) + 1
    return int(area_km2 * (windows_per_side**2))


def build_pilot_summary(config: PilotConfig) -> dict[str, object]:
    """Build a summary object for a local pilot run."""

    areas: list[dict[str, object]] = []
    total_area_km2 = 0.0
    total_raw_mb = 0.0
    total_non_overlapping_tiles = 0
    total_windows = 0

    for city in config.cities:
        city_raw_mb = raw_size_mb(city.area_km2)
        city_tiles = non_overlapping_tile_count(city.area_km2, config.tile_size_m)
        city_windows = sliding_window_count(
            city.area_km2,
            config.tile_size_m,
            config.sliding_window_step_m,
        )
        total_area_km2 += city.area_km2
        total_raw_mb += city_raw_mb
        total_non_overlapping_tiles += city_tiles
        total_windows += city_windows
        areas.append(
            {
                **asdict(city),
                "estimated_raw_mb": round(city_raw_mb, 1),
                "estimated_non_overlapping_tiles": city_tiles,
                "estimated_windows": city_windows,
            }
        )

    max_raw_mb = config.max_raw_size_gb * 1_000
    return {
        "run_name": config.run_name,
        "imagery_resolution_cm": config.imagery_resolution_cm,
        "tile_size_m": config.tile_size_m,
        "sliding_window_step_m": config.sliding_window_step_m,
        "target_per_class": config.target_per_class,
        "split_targets": derive_split_targets(config.target_per_class, config.split_ratios),
        "wmts_zoom": config.wmts_zoom,
        "wmts_tile_radius": config.wmts_tile_radius,
        "auto_search_radius": config.auto_search_radius,
        "areas": areas,
        "totals": {
            "area_km2": round(total_area_km2, 1),
            "raw_mb": round(total_raw_mb, 1),
            "raw_gb": round(total_raw_mb / 1_000, 3),
            "non_overlapping_tiles": total_non_overlapping_tiles,
            "windows": total_windows,
            "max_raw_gb": config.max_raw_size_gb,
            "within_budget": total_raw_mb <= max_raw_mb,
        },
    }


def repo_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[3]


def write_default_config(destination: Path) -> Path:
    """Write the tracked local debug config to a destination."""

    source = repo_root() / "configs" / "local-debug.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text())
    return destination


def bootstrap_local_pilot(config_path: Path) -> dict[str, Path]:
    """Create local run folders and write planning artifacts."""

    config = load_pilot_config(config_path)
    summary = build_pilot_summary(config)

    root = repo_root()
    raw_root = root / "data" / "raw" / config.run_name
    processed_root = root / "data" / "processed" / config.run_name
    manifests_root = processed_root / "manifests"
    reports_root = processed_root / "reports"
    tiles_root = processed_root / "tiles"
    weak_labels_root = processed_root / "weak-labels"
    reviews_root = processed_root / "reviews"
    exports_root = processed_root / "exports"

    for directory in (
        raw_root,
        manifests_root,
        reports_root,
        tiles_root,
        weak_labels_root,
        reviews_root,
        exports_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    summary_json = manifests_root / "pilot-plan.json"
    summary_json.write_text(json.dumps(summary, indent=2))

    areas_csv = manifests_root / "areas.csv"
    with areas_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "country",
                "split",
                "area_km2",
                "estimated_raw_mb",
                "estimated_non_overlapping_tiles",
                "estimated_windows",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(summary["areas"])

    report_md = reports_root / "pilot-plan.md"
    report_md.write_text(render_pilot_report(summary))

    next_steps = reports_root / "next-steps.txt"
    next_steps.write_text(
        "\n".join(
            [
                "1. Fetch raw imagery for the listed city areas.",
                "2. Save imagery under data/raw/<run_name>/.",
                "3. Generate 25 m x 25 m tiles into data/processed/<run_name>/tiles/.",
                "4. Write weak labels into data/processed/<run_name>/weak-labels/.",
                "5. Review uncertain cases in data/processed/<run_name>/reviews/.",
                "6. Export the first training set into data/processed/<run_name>/exports/.",
            ]
        )
        + "\n"
    )

    return {
        "config": config_path,
        "raw_root": raw_root,
        "processed_root": processed_root,
        "summary_json": summary_json,
        "areas_csv": areas_csv,
        "report_md": report_md,
        "next_steps": next_steps,
    }


def lat_lon_to_xyz(latitude: float, longitude: float, zoom: int) -> tuple[int, int]:
    """Convert latitude and longitude to XYZ tile coordinates."""

    lat_rad = math.radians(latitude)
    tile_count = 2**zoom
    tile_col = int((longitude + 180.0) / 360.0 * tile_count)
    tile_row = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * tile_count)
    return tile_col, tile_row


def iter_city_tile_jobs(
    city: CityPlan,
    zoom: int,
    tile_radius: int,
) -> list[dict[str, int | str]]:
    """Return a square WMTS fetch grid around a city center."""

    center_col, center_row = lat_lon_to_xyz(city.latitude, city.longitude, zoom)
    jobs: list[dict[str, int | str]] = []
    for col in range(center_col - tile_radius, center_col + tile_radius + 1):
        for row in range(center_row - tile_radius, center_row + tile_radius + 1):
            jobs.append(
                {
                    "city": city.name,
                    "split": city.split,
                    "zoom": zoom,
                    "tile_col": col,
                    "tile_row": row,
                }
            )
    return jobs


def fetch_wmts_debug_tiles(config_path: Path) -> dict[str, object]:
    """Download a small WMTS tile grid for local debugging."""

    config = load_pilot_config(config_path)
    root = repo_root()
    raw_root = root / "data" / "raw" / config.run_name / "wmts-3857-z20"
    manifests_root = root / "data" / "processed" / config.run_name / "manifests"
    raw_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, int | str]] = []
    downloaded = 0
    total_bytes = 0

    for city in config.cities:
        city_root = raw_root / city.name.lower().replace(" ", "-")
        city_root.mkdir(parents=True, exist_ok=True)
        for job in iter_city_tile_jobs(city, config.wmts_zoom, config.wmts_tile_radius):
            destination = (
                city_root
                / str(job["zoom"])
                / str(job["tile_col"])
                / f"{job['tile_row']}.jpeg"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = WMTS_TEMPLATE.format(
                zoom=job["zoom"],
                tile_col=job["tile_col"],
                tile_row=job["tile_row"],
            )
            if not destination.exists():
                with urlopen(url) as response:
                    payload = response.read()
                destination.write_bytes(payload)
            size_bytes = destination.stat().st_size
            total_bytes += size_bytes
            downloaded += 1
            manifest_rows.append(
                {
                    **job,
                    "url": url,
                    "path": str(destination),
                    "size_bytes": size_bytes,
                }
            )

    manifest_csv = manifests_root / "wmts-debug-tiles.csv"
    with manifest_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "city",
                "split",
                "zoom",
                "tile_col",
                "tile_row",
                "url",
                "path",
                "size_bytes",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "run_name": config.run_name,
        "zoom": config.wmts_zoom,
        "tile_radius": config.wmts_tile_radius,
        "downloaded_tiles": downloaded,
        "total_size_bytes": total_bytes,
        "total_size_mb": round(total_bytes / 1_000_000, 3),
        "raw_root": str(raw_root),
        "manifest_csv": str(manifest_csv),
    }
    summary_json = manifests_root / "wmts-debug-summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    return summary


def _download_job(job: dict[str, int | str], raw_root: Path) -> dict[str, int | str]:
    city_root = raw_root / str(job["city"]).lower().replace(" ", "-")
    destination = city_root / str(job["zoom"]) / str(job["tile_col"]) / f"{job['tile_row']}.jpeg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = WMTS_TEMPLATE.format(
        zoom=job["zoom"],
        tile_col=job["tile_col"],
        tile_row=job["tile_row"],
    )
    if not destination.exists():
        with urlopen(url) as response:
            payload = response.read()
        destination.write_bytes(payload)
    return {
        **job,
        "url": url,
        "path": str(destination),
        "size_bytes": destination.stat().st_size,
    }


def fetch_wmts_search_tiles(config_path: Path, radius: int | None = None) -> dict[str, object]:
    """Download a wider cached WMTS search pool around each city center."""

    config = load_pilot_config(config_path)
    search_radius = radius if radius is not None else config.auto_search_radius
    root = repo_root()
    raw_root = root / "data" / "raw" / config.run_name / "wmts-3857-z20"
    manifests_root = root / "data" / "processed" / config.run_name / "manifests"
    raw_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, int | str]] = []
    for city in config.cities:
        jobs.extend(iter_city_tile_jobs(city, config.wmts_zoom, search_radius))

    results: list[dict[str, int | str]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_download_job, job, raw_root) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda row: (str(row["city"]), int(row["tile_col"]), int(row["tile_row"])))
    manifest_csv = manifests_root / "wmts-search-tiles.csv"
    with manifest_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "city",
                "split",
                "zoom",
                "tile_col",
                "tile_row",
                "url",
                "path",
                "size_bytes",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    total_bytes = sum(int(row["size_bytes"]) for row in results)
    summary = {
        "run_name": config.run_name,
        "zoom": config.wmts_zoom,
        "search_radius": search_radius,
        "downloaded_tiles": len(results),
        "total_size_bytes": total_bytes,
        "total_size_mb": round(total_bytes / 1_000_000, 3),
        "raw_root": str(raw_root),
        "manifest_csv": str(manifest_csv),
    }
    summary_json = manifests_root / "wmts-search-summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    return summary


def _load_rgb_arrays():
    from PIL import Image  # type: ignore
    import numpy as np  # type: ignore

    return Image, np


def _load_pil_augmenters():
    from PIL import Image, ImageEnhance, ImageOps  # type: ignore

    return Image, ImageEnhance, ImageOps


def extract_crosswalk_features(image_path: Path) -> dict[str, float]:
    """Extract a compact feature vector from one tile."""

    Image, np = _load_rgb_arrays()
    from collections import deque

    image = Image.open(image_path).convert("HSV")
    hsv = np.array(image)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yellow = ((h >= 18) & (h <= 55) & (s >= 40) & (v >= 100)).astype(np.uint8)

    seen = np.zeros_like(yellow, dtype=bool)
    component_count = 0
    medium_components = 0
    medium_area = 0
    elongated_components = 0
    large_components = 0

    height, width = yellow.shape
    for y in range(height):
        for x in range(width):
            if yellow[y, x] == 0 or seen[y, x]:
                continue
            component_count += 1
            queue = deque([(y, x)])
            seen[y, x] = True
            size = 0
            min_x = max_x = x
            min_y = max_y = y
            while queue:
                cy, cx = queue.popleft()
                size += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and yellow[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))

            box_width = max_x - min_x + 1
            box_height = max_y - min_y + 1
            aspect_ratio = max(box_width, box_height) / max(1, min(box_width, box_height))

            if size > 800:
                large_components += 1
            if 35 <= size <= 450 and aspect_ratio >= 1.35:
                medium_components += 1
                medium_area += size
            if 20 <= size <= 450 and aspect_ratio >= 2.3:
                elongated_components += 1

    density = float(yellow.mean())
    row_profile = yellow.mean(axis=1)
    col_profile = yellow.mean(axis=0)
    stripe_axis_std = float(max(row_profile.std(), col_profile.std()))
    stripe_axis_peak = float(max(np.sort(row_profile)[-6:].sum(), np.sort(col_profile)[-6:].sum()))

    return {
        "yellow_density": density,
        "component_count": float(component_count),
        "medium_component_count": float(medium_components),
        "medium_component_area": float(medium_area),
        "elongated_component_count": float(elongated_components),
        "large_component_count": float(large_components),
        "stripe_axis_std": stripe_axis_std,
        "stripe_axis_peak": stripe_axis_peak,
    }


def crosswalk_score(image_path: Path) -> float:
    """Heuristic score for likely crosswalk tiles."""

    features = extract_crosswalk_features(image_path)
    score = (
        (features["medium_component_count"] * 3.5)
        + (features["medium_component_area"] / 220.0)
        + (features["elongated_component_count"] * 2.5)
        + (features["stripe_axis_std"] * 60.0)
        + (features["stripe_axis_peak"] * 8.0)
        - (max(0.0, features["yellow_density"] - 0.12) * 180.0)
        - (features["large_component_count"] * 4.0)
    )
    return round(score, 4)


def _feature_vector(features: dict[str, float], feature_names: tuple[str, ...]) -> tuple[float, ...]:
    return tuple(features[name] for name in feature_names)


def train_centroid_model(review_csv: Path, raw_root: Path) -> CrosswalkCentroidModel | None:
    """Train a tiny distance-based classifier from reviewed bootstrap labels."""

    if not review_csv.exists():
        return None

    feature_names = (
        "yellow_density",
        "medium_component_count",
        "medium_component_area",
        "elongated_component_count",
        "large_component_count",
        "stripe_axis_std",
        "stripe_axis_peak",
    )
    grouped: dict[str, list[tuple[float, ...]]] = {"crosswalk": [], "no_crosswalk": []}
    seen_paths: set[str] = set()

    with review_csv.open() as handle:
        for row in csv.DictReader(handle):
            label = row["label"]
            if label not in grouped:
                continue
            rel = row["relative_path"]
            if rel in seen_paths:
                continue
            seen_paths.add(rel)
            path = raw_root / rel
            if not path.exists():
                continue
            features = extract_crosswalk_features(path)
            grouped[label].append(_feature_vector(features, feature_names))

    if not grouped["crosswalk"] or not grouped["no_crosswalk"]:
        return None

    _, np = _load_rgb_arrays()
    pos = np.array(grouped["crosswalk"], dtype=float)
    neg = np.array(grouped["no_crosswalk"], dtype=float)
    combined = np.vstack([pos, neg])
    scale = combined.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)

    return CrosswalkCentroidModel(
        feature_names=feature_names,
        positive_mean=tuple(pos.mean(axis=0)),
        negative_mean=tuple(neg.mean(axis=0)),
        scale=tuple(scale),
        training_samples=int(pos.shape[0] + neg.shape[0]),
    )


def predict_crosswalk_probability(
    image_path: Path,
    model: CrosswalkCentroidModel | None,
) -> tuple[float, float]:
    """Predict a crosswalk probability and score for one tile."""

    if model is None:
        score = crosswalk_score(image_path)
        probability = 1.0 / (1.0 + math.exp(-(score / 6.0)))
        return probability, score

    _, np = _load_rgb_arrays()
    features = extract_crosswalk_features(image_path)
    vector = np.array(_feature_vector(features, model.feature_names), dtype=float)
    pos_mean = np.array(model.positive_mean, dtype=float)
    neg_mean = np.array(model.negative_mean, dtype=float)
    scale = np.array(model.scale, dtype=float)

    normalized = vector / scale
    pos_distance = np.linalg.norm(normalized - (pos_mean / scale))
    neg_distance = np.linalg.norm(normalized - (neg_mean / scale))
    score = float(neg_distance - pos_distance)
    probability = 1.0 / (1.0 + math.exp(-(score * 2.0)))
    return probability, score


def _load_review_overrides(review_csv: Path) -> dict[str, dict[str, str]]:
    overrides: dict[str, dict[str, str]] = {}
    if not review_csv.exists():
        return overrides
    with review_csv.open() as handle:
        for row in csv.DictReader(handle):
            overrides[row["relative_path"]] = row
    return overrides


def build_full_capped_dataset(
    config_path: Path,
    review_csv: Path,
    export_name: str = "full-capped-v1",
    search_radius: int | None = None,
) -> dict[str, object]:
    """Build a capped dataset from unique raw images only."""

    config = load_pilot_config(config_path)
    if search_radius is not None:
        fetch_wmts_search_tiles(config_path, search_radius)

    split_targets = derive_split_targets(config.target_per_class, config.split_ratios)
    raw_root = repo_root() / "data" / "raw" / config.run_name / "wmts-3857-z20"
    export_root = repo_root() / "data" / "processed" / config.run_name / "exports" / export_name
    image_root = export_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    model = train_centroid_model(review_csv, raw_root)
    all_rows: list[dict[str, str | float]] = []
    for path in sorted(raw_root.rglob("*.jpeg")):
        rel = path.relative_to(raw_root).as_posix()
        city = rel.split("/")[0]
        split = next(city_plan.split for city_plan in config.cities if city_plan.name.lower().replace(" ", "-") == city)
        probability, score = predict_crosswalk_probability(path, model)
        all_rows.append(
            {
                "relative_path": rel,
                "city": city,
                "split": split,
                "label": "",
                "confidence": round(probability, 4),
                "review_status": "auto-labeled",
                "label_source": "model-centroid-v1" if model is not None else "heuristic-v2",
                "note": "assigned automatically from unique raw imagery",
                "score": score,
                "probability_crosswalk": probability,
            }
        )

    rows_by_split = defaultdict(list)
    for row in all_rows:
        rows_by_split[str(row["split"])].append(row)

    selected_rows: list[dict[str, str | float]] = []
    actual_split_caps: dict[str, int] = {}
    per_split_class_counts: dict[str, dict[str, int]] = {}
    available_raw_per_split: dict[str, int] = {}
    available_raw_total = len(all_rows)

    for split, target in split_targets.items():
        split_rows = rows_by_split.get(split, [])
        available_raw_per_split[split] = len(split_rows)
        actual_cap = min(target, len(split_rows) // 2)
        actual_split_caps[split] = actual_cap

        positive_candidates = sorted(
            split_rows,
            key=lambda row: float(row["probability_crosswalk"]),
            reverse=True,
        )
        final_pos = positive_candidates[:actual_cap]
        positive_paths = {str(row["relative_path"]) for row in final_pos}
        negative_candidates = [
            row
            for row in sorted(split_rows, key=lambda row: float(row["probability_crosswalk"]))
            if str(row["relative_path"]) not in positive_paths
        ]
        final_neg = negative_candidates[:actual_cap]

        for row in final_pos:
            row["label"] = "crosswalk"
            row["review_status"] = "auto-positive"
            row["note"] = "selected automatically from the highest crosswalk probabilities"
        for row in final_neg:
            row["label"] = "no_crosswalk"
            row["review_status"] = "auto-negative"
            row["note"] = "selected automatically from the lowest crosswalk probabilities"

        selected_rows.extend(final_pos)
        selected_rows.extend(final_neg)
        per_split_class_counts[split] = {
            "crosswalk": len(final_pos),
            "no_crosswalk": len(final_neg),
        }

    class_counts = defaultdict(int)
    for row in selected_rows:
        class_counts[str(row["label"])] += 1

    for row in selected_rows:
        source = raw_root / str(row["relative_path"])
        destination = image_root / str(row["label"]) / str(row["relative_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        row["export_path"] = str(destination)

    manifest_csv = export_root / "labels.csv"
    with manifest_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "relative_path",
                "city",
                "split",
                "label",
                "confidence",
                "review_status",
                "label_source",
                "note",
                "score",
                "probability_crosswalk",
                "export_path",
            ],
        )
        writer.writeheader()
        writer.writerows(selected_rows)

    summary = {
        "export_name": export_name,
        "target_per_class": config.target_per_class,
        "split_targets": split_targets,
        "actual_split_caps": actual_split_caps,
        "positive_count": class_counts.get("crosswalk", 0),
        "negative_count": class_counts.get("no_crosswalk", 0),
        "total_count": len(selected_rows),
        "per_split_class_counts": per_split_class_counts,
        "available_raw_total": available_raw_total,
        "available_raw_per_split": available_raw_per_split,
        "model_name": "model-centroid-v1" if model is not None else "heuristic-v2",
        "training_samples": 0 if model is None else model.training_samples,
        "is_target_full": all(
            per_split_class_counts.get(split, {}).get(label, 0) == target
            for split, target in split_targets.items()
            for label in ("crosswalk", "no_crosswalk")
        ),
        "manifest_csv": str(manifest_csv),
        "image_root": str(image_root),
    }
    summary_json = export_root / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    return summary


def export_balanced_seed_set(
    review_csv: Path,
    export_name: str = "balanced-seed-v1",
    target_per_class: int | None = None,
    split_targets: dict[str, int] | None = None,
) -> dict[str, object]:
    """Export a balanced capped subset with class and split caps."""

    rows: list[dict[str, str]] = []
    with review_csv.open() as handle:
        rows = list(csv.DictReader(handle))

    by_label_and_split: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    labels = sorted({row["label"] for row in rows})
    splits = sorted({row["split"] for row in rows})
    for row in rows:
        by_label_and_split[(row["label"], row["split"])].append(row)

    if split_targets is None:
        if target_per_class is None:
            available_per_split = {
                split: min(
                    len(by_label_and_split[("crosswalk", split)]),
                    len(by_label_and_split[("no_crosswalk", split)]),
                )
                for split in splits
            }
            split_targets = available_per_split
        else:
            split_targets = {split: 0 for split in splits}
            split_targets[splits[0]] = target_per_class

    actual_split_caps = {
        split: min(
            split_targets.get(split, 0),
            *(len(by_label_and_split[(label, split)]) for label in labels),
        )
        for split in split_targets
    }

    selected_rows: list[dict[str, str]] = []
    for split, cap in actual_split_caps.items():
        for label in labels:
            selected_rows.extend(by_label_and_split[(label, split)][:cap])

    class_counts: dict[str, int] = defaultdict(int)
    per_split_class_counts: dict[str, dict[str, int]] = {}
    for split in split_targets:
        per_split_class_counts[split] = {}
        for label in labels:
            count = len([row for row in selected_rows if row["split"] == split and row["label"] == label])
            per_split_class_counts[split][label] = count
            class_counts[label] += count

    root = repo_root()
    raw_root = root / "data" / "raw" / "local-debug-v1" / "wmts-3857-z20"
    export_root = root / "data" / "processed" / "local-debug-v1" / "exports" / export_name
    image_root = export_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    for row in selected_rows:
        source = raw_root / row["relative_path"]
        destination = image_root / row["label"] / row["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        row["export_path"] = str(destination)

    manifest_csv = export_root / "labels.csv"
    with manifest_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "relative_path",
                "city",
                "split",
                "label",
                "confidence",
                "review_status",
                "label_source",
                "note",
                "export_path",
            ],
        )
        writer.writeheader()
        writer.writerows(selected_rows)

    summary = {
        "export_name": export_name,
        "target_per_class": target_per_class,
        "split_targets": split_targets,
        "actual_split_caps": actual_split_caps,
        "available_counts": {
            split: {
                label: len(by_label_and_split[(label, split)]) for label in labels
            }
            for split in split_targets
        },
        "positive_count": class_counts.get("crosswalk", 0),
        "negative_count": class_counts.get("no_crosswalk", 0),
        "total_count": len(selected_rows),
        "per_split_class_counts": per_split_class_counts,
        "is_target_full": (
            target_per_class is not None
            and all(actual_split_caps.get(split, 0) == split_targets.get(split, 0) for split in split_targets)
        ),
        "manifest_csv": str(manifest_csv),
        "image_root": str(image_root),
    }
    summary_json = export_root / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    return summary


def render_pilot_report(summary: dict[str, object]) -> str:
    """Render a short markdown report for the current pilot plan."""

    totals = summary["totals"]
    areas = summary["areas"]
    lines = [
        f"# Pilot Plan: {summary['run_name']}",
        "",
        "## Budget",
        "",
        f"- Raw imagery budget cap: `{totals['max_raw_gb']} GB`",
        f"- Estimated raw imagery: `{totals['raw_gb']} GB`",
        f"- Within budget: `{totals['within_budget']}`",
        f"- Current target per class: `{summary['target_per_class']}`",
        f"- Split targets per class: `{summary['split_targets']}`",
        f"- WMTS zoom for the first local fetch: `{summary['wmts_zoom']}`",
        f"- WMTS tile radius per city: `{summary['wmts_tile_radius']}`",
        f"- Auto search radius for the full run: `{summary['auto_search_radius']}`",
        "",
        "## Totals",
        "",
        f"- Total search area: `{totals['area_km2']} km²`",
        f"- Estimated non-overlapping task tiles: `{totals['non_overlapping_tiles']}`",
        f"- Estimated windows at current step size: `{totals['windows']}`",
        "",
        "## Areas",
        "",
    ]
    for area in areas:
        lines.extend(
            [
                f"### {area['name']}",
                "",
                f"- Split: `{area['split']}`",
                f"- Search area: `{area['area_km2']} km²`",
                f"- Estimated raw imagery: `{area['estimated_raw_mb']} MB`",
                f"- Estimated non-overlapping task tiles: `{area['estimated_non_overlapping_tiles']}`",
                f"- Estimated windows: `{area['estimated_windows']}`",
                f"- Notes: {area['notes']}",
                "",
            ]
        )
    return "\n".join(lines)

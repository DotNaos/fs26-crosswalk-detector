from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

EARTH_RADIUS_M = 6_378_137.0
TILE_SIZE_M = 25.0
IMAGE_PX = 256
WMS_BASE_URL = "https://wms.geo.admin.ch/"
OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
USER_AGENT = "crosswalk-detector-fs26/0.1"


@dataclass(frozen=True)
class CityBox:
    name: str
    split: str
    bbox: tuple[float, float, float, float]


CITY_BOXES = (
    CityBox("zurich", "train", (47.30, 8.45, 47.44, 8.62)),
    CityBox("basel", "train", (47.50, 7.50, 47.61, 7.68)),
    CityBox("bern", "train", (46.90, 7.35, 47.00, 7.52)),
    CityBox("geneva", "train", (46.16, 6.08, 46.24, 6.18)),
    CityBox("lausanne", "train", (46.49, 6.58, 46.55, 6.69)),
    CityBox("winterthur", "val", (47.46, 8.68, 47.53, 8.78)),
    CityBox("lucerne", "val", (47.02, 8.25, 47.08, 8.36)),
    CityBox("st-gallen", "val", (47.39, 9.32, 47.45, 9.43)),
    CityBox("biel", "val", (47.11, 7.20, 47.17, 7.30)),
    CityBox("thun", "val", (46.72, 7.58, 46.78, 7.66)),
    CityBox("lugano", "test", (45.98, 8.90, 46.04, 8.99)),
    CityBox("chur", "test", (46.82, 9.49, 46.88, 9.57)),
    CityBox("fribourg", "test", (46.77, 7.12, 46.84, 7.20)),
    CityBox("schaffhausen", "test", (47.67, 8.58, 47.73, 8.70)),
    CityBox("neuchatel", "test", (46.98, 6.88, 47.03, 6.98)),
    CityBox("sion", "test", (46.21, 7.31, 46.26, 7.39)),
    CityBox("zug", "test", (47.14, 8.47, 47.20, 8.56)),
    CityBox("aarau", "test", (47.36, 8.00, 47.42, 8.09)),
    CityBox("solothurn", "test", (47.18, 7.49, 47.23, 7.56)),
    CityBox("rapperswil", "test", (47.20, 8.78, 47.25, 8.85)),
    CityBox("baden", "test", (47.44, 8.25, 47.50, 8.35)),
    CityBox("wil", "test", (47.44, 9.00, 47.49, 9.08)),
    CityBox("nyon", "test", (46.36, 6.19, 46.41, 6.26)),
    CityBox("vevey", "test", (46.44, 6.80, 46.49, 6.88)),
    CityBox("montreux", "test", (46.42, 6.88, 46.47, 6.96)),
    CityBox("yverdon", "test", (46.75, 6.60, 46.81, 6.68)),
    CityBox("bellinzona", "test", (46.17, 8.96, 46.22, 9.05)),
    CityBox("locarno", "test", (46.14, 8.76, 46.19, 8.83)),
    CityBox("kreuzlingen", "test", (47.62, 9.13, 47.68, 9.22)),
    CityBox("olten", "test", (47.32, 7.86, 47.37, 7.95)),
)


def mercator_from_lat_lon(latitude: float, longitude: float) -> tuple[float, float]:
    x = EARTH_RADIUS_M * math.radians(longitude)
    y = EARTH_RADIUS_M * math.log(math.tan(math.pi / 4.0 + math.radians(latitude) / 2.0))
    return x, y


def lat_lon_from_mercator(x: float, y: float) -> tuple[float, float]:
    longitude = math.degrees(x / EARTH_RADIUS_M)
    latitude = math.degrees(2.0 * math.atan(math.exp(y / EARTH_RADIUS_M)) - math.pi / 2.0)
    return latitude, longitude


def tile_bbox(x: float, y: float) -> tuple[float, float, float, float]:
    half = TILE_SIZE_M / 2.0
    return x - half, y - half, x + half, y + half


def wms_url(bbox: tuple[float, float, float, float]) -> str:
    query = urlencode(
        {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": "ch.swisstopo.swissimage-product",
            "STYLES": "default",
            "CRS": "EPSG:3857",
            "BBOX": ",".join(f"{value:.3f}" for value in bbox),
            "WIDTH": IMAGE_PX,
            "HEIGHT": IMAGE_PX,
            "FORMAT": "image/jpeg",
        }
    )
    return f"{WMS_BASE_URL}?{query}"


def fetch_crossings(city: CityBox, cache_root: Path) -> list[dict[str, object]]:
    cache_path = cache_root / f"{city.name}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    south, west, north, east = city.bbox
    query = f'[out:json][timeout:90];node["highway"="crossing"]({south},{west},{north},{east});out body;'
    payload = None
    last_error: Exception | None = None
    for attempt in range(5):
        for endpoint in OVERPASS_URLS:
            request = Request(
                endpoint,
                data=urlencode({"data": query}).encode(),
                headers={"User-Agent": USER_AGENT},
            )
            try:
                with urlopen(request, timeout=120) as response:
                    payload = json.load(response)
                break
            except Exception as exc:
                last_error = exc
        if payload is not None:
            break
        time.sleep(4.0 * (attempt + 1))
    if payload is None:
        raise RuntimeError(f"Failed to fetch OSM crossings for {city.name}: {last_error}")
    elements = payload.get("elements", [])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(elements, separators=(",", ":")))
    return elements


def split_targets(target_per_class: int) -> dict[str, int]:
    return {
        "train": int(target_per_class * 0.7),
        "val": int(target_per_class * 0.15),
        "test": target_per_class - int(target_per_class * 0.7) - int(target_per_class * 0.15),
    }


def positive_rows(cities: tuple[CityBox, ...], target_per_class: int, cache_root: Path) -> list[dict[str, object]]:
    targets = split_targets(target_per_class)
    rows: list[dict[str, object]] = []
    rng = random.Random(42)
    for split, target in targets.items():
        candidates = []
        for city in cities:
            if city.split != split:
                continue
            for element in fetch_crossings(city, cache_root):
                lat = float(element["lat"])
                lon = float(element["lon"])
                x, y = mercator_from_lat_lon(lat, lon)
                candidates.append(
                    {
                        "osm_id": str(element["id"]),
                        "city": city.name,
                        "split": split,
                        "label": "crosswalk",
                        "latitude": lat,
                        "longitude": lon,
                        "center_x": x,
                        "center_y": y,
                    }
                )
        rng.shuffle(candidates)
        if len(candidates) < target:
            raise RuntimeError(f"Only found {len(candidates)} OSM crossings for {split}, need {target}.")
        rows.extend(candidates[:target])
    return rows


def negative_rows(positives: list[dict[str, object]], target_per_class: int) -> list[dict[str, object]]:
    targets = split_targets(target_per_class)
    rng = random.Random(84)
    positives_by_split: dict[str, list[dict[str, object]]] = {}
    for row in positives:
        positives_by_split.setdefault(str(row["split"]), []).append(row)

    rows: list[dict[str, object]] = []
    for split, target in targets.items():
        positive_split = positives_by_split[split]
        positive_points = [(float(row["center_x"]), float(row["center_y"])) for row in positive_split]
        cell_size = 55.0
        positive_grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for px, py in positive_points:
            positive_grid.setdefault((int(px // cell_size), int(py // cell_size)), []).append((px, py))
        split_rows: list[dict[str, object]] = []
        attempts = 0
        while len(split_rows) < target:
            attempts += 1
            if attempts > target * 80:
                raise RuntimeError(f"Could not sample enough negatives for {split}.")
            anchor = rng.choice(positive_split)
            angle = rng.random() * math.tau
            distance = rng.uniform(75.0, 230.0)
            x = float(anchor["center_x"]) + math.cos(angle) * distance
            y = float(anchor["center_y"]) + math.sin(angle) * distance
            cx = int(x // cell_size)
            cy = int(y // cell_size)
            nearby_points = [
                point
                for gx in range(cx - 1, cx + 2)
                for gy in range(cy - 1, cy + 2)
                for point in positive_grid.get((gx, gy), [])
            ]
            if nearby_points and min(math.hypot(x - px, y - py) for px, py in nearby_points) < 45.0:
                continue
            lat, lon = lat_lon_from_mercator(x, y)
            split_rows.append(
                {
                    "osm_id": "",
                    "city": anchor["city"],
                    "split": split,
                    "label": "no_crosswalk",
                    "latitude": lat,
                    "longitude": lon,
                    "center_x": x,
                    "center_y": y,
                }
            )
        rows.extend(split_rows)
    return rows


def download_tile(row: dict[str, object], destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = wms_url(tile_bbox(float(row["center_x"]), float(row["center_y"])))
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=45) as response:
                destination.write_bytes(response.read())
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Failed to download {destination}: {last_error}")


def materialize_dataset(rows: list[dict[str, object]], export_root: Path, workers: int) -> list[dict[str, object]]:
    image_root = export_root / "images"
    indexed_rows = []
    for index, row in enumerate(rows):
        tile_id = f"{row['split']}:{row['label']}:{index:05d}"
        bbox = tile_bbox(float(row["center_x"]), float(row["center_y"]))
        relative_path = f"{row['split']}/{row['label']}/{tile_id.replace(':', '-')}.jpg"
        image_path = image_root / relative_path
        indexed_rows.append(
            {
                **row,
                "tile_id": tile_id,
                "bbox_mercator": bbox,
                "relative_path": relative_path,
                "image_path": str(image_path),
            }
        )

    started = time.monotonic()
    total = len(indexed_rows)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_tile, row, Path(str(row["image_path"]))) for row in indexed_rows]
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            if completed == total or completed % 500 == 0:
                elapsed = max(0.1, time.monotonic() - started)
                rate = completed / elapsed
                print(f"downloaded={completed}/{total} rate={rate:.1f}/s", flush=True)
    return indexed_rows


def write_outputs(rows: list[dict[str, object]], export_root: Path, run_name: str, export_name: str) -> None:
    export_root.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tile_id",
        "split",
        "label",
        "city",
        "osm_id",
        "latitude",
        "longitude",
        "center_x",
        "center_y",
        "min_x",
        "min_y",
        "max_x",
        "max_y",
        "relative_path",
        "image_path",
    ]
    with (export_root / "labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            min_x, min_y, max_x, max_y = row["bbox_mercator"]
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames if key in row},
                    "min_x": round(float(min_x), 3),
                    "min_y": round(float(min_y), 3),
                    "max_x": round(float(max_x), 3),
                    "max_y": round(float(max_y), 3),
                }
            )

    compact = {
        "run_name": run_name,
        "export_name": export_name,
        "format": "crosswalk-osm-compact-v1",
        "source": "OSM highway=crossing positives, offset negatives, swisstopo WMS imagery",
        "tile_size_m": TILE_SIZE_M,
        "image_px": IMAGE_PX,
        "tiles": [
            {
                "tile_id": row["tile_id"],
                "split": row["split"],
                "label": row["label"],
                "city": row["city"],
                "osm_id": row["osm_id"],
                "latitude": round(float(row["latitude"]), 7),
                "longitude": round(float(row["longitude"]), 7),
                "bbox_mercator": [round(float(value), 3) for value in row["bbox_mercator"]],
            }
            for row in rows
        ],
    }
    with gzip.open(export_root / "compact-manifest.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(compact, handle, separators=(",", ":"))

    with gzip.open(export_root / "labels.compact.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["tile_id", "split", "label", "city", "osm_id", "latitude", "longitude", "min_x", "min_y", "max_x", "max_y"],
        )
        writer.writeheader()
        for row in rows:
            min_x, min_y, max_x, max_y = row["bbox_mercator"]
            writer.writerow(
                {
                    "tile_id": row["tile_id"],
                    "split": row["split"],
                    "label": row["label"],
                    "city": row["city"],
                    "osm_id": row["osm_id"],
                    "latitude": round(float(row["latitude"]), 7),
                    "longitude": round(float(row["longitude"]), 7),
                    "min_x": round(float(min_x), 3),
                    "min_y": round(float(min_y), 3),
                    "max_x": round(float(max_x), 3),
                    "max_y": round(float(max_y), 3),
                }
            )

    counts: dict[str, int] = {}
    splits: dict[str, dict[str, int]] = {}
    for row in rows:
        counts[str(row["label"])] = counts.get(str(row["label"]), 0) + 1
        splits.setdefault(str(row["split"]), {}).setdefault(str(row["label"]), 0)
        splits[str(row["split"])][str(row["label"])] += 1
    summary = {
        "run_name": run_name,
        "export_name": export_name,
        "total": len(rows),
        "counts": counts,
        "splits": splits,
        "tile_size_m": TILE_SIZE_M,
        "image_px": IMAGE_PX,
        "missing_images": sum(1 for row in rows if not Path(str(row["image_path"])).exists()),
        "labels_csv": str(export_root / "labels.csv"),
        "compact_json": str(export_root / "compact-manifest.json.gz"),
        "compact_csv": str(export_root / "labels.compact.csv.gz"),
    }
    (export_root / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="osm-v1-10k")
    parser.add_argument("--export-name", default="balanced-10k-v1")
    parser.add_argument("--target-per-class", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    export_root = repo_root / "data" / "processed" / args.run_name / "exports" / args.export_name
    cache_root = repo_root / "data" / "raw" / args.run_name / "overpass"
    positives = positive_rows(CITY_BOXES, args.target_per_class, cache_root)
    negatives = negative_rows(positives, args.target_per_class)
    rows = positives + negatives
    rows.sort(key=lambda row: (str(row["split"]), str(row["label"]), str(row["city"]), str(row["osm_id"])))
    materialized = materialize_dataset(rows, export_root, args.workers)
    write_outputs(materialized, export_root, args.run_name, args.export_name)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

from PIL import Image

from crosswalk_detector.urban_vision import _dilate, build_urban_road_density_mask


def build_road_channel_export(source_export: Path, output_root: Path, *, road_threshold: float, road_dilate: int) -> dict:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    for folder in ("images", "masks"):
        shutil.copytree(source_export / folder, output_root / folder, symlinks=True)

    rows = _load_rows(source_export / "manifest.csv")
    fieldnames = list(rows[0].keys())
    if "road_path" not in fieldnames:
        fieldnames.append("road_path")

    road_cache: dict[str, str] = {}
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            road_path = road_cache.get(row["image_path"])
            if road_path is None:
                road_path = _write_road_mask(source_export, output_root, row["image_path"], road_threshold, road_dilate)
                road_cache[row["image_path"]] = road_path
            writer.writerow({**row, "road_path": road_path})

    source_summary = json.loads((source_export / "summary.json").read_text(encoding="utf8"))
    summary = {
        "source_export": str(source_export),
        "samples": len(rows),
        "unique_road_masks": len(road_cache),
        "road_threshold": road_threshold,
        "road_dilate": road_dilate,
        "source_summary": source_summary,
        "counts": _count_manifest(output_root / "manifest.csv"),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    return summary


def _write_road_mask(source_export: Path, output_root: Path, image_path: str, road_threshold: float, road_dilate: int) -> str:
    image = Image.open(source_export / image_path).convert("RGB")
    result = build_urban_road_density_mask(image, max_width=image.width, threshold=road_threshold)
    mask = result.mask.astype(bool)
    if road_dilate > 0:
        mask = _dilate(mask, iterations=road_dilate)
    target = output_root / "road_masks" / Path(image_path).with_suffix(".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype("uint8") * 255).save(target)
    return target.relative_to(output_root).as_posix()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _count_manifest(path: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf8") as handle:
        for row in csv.DictReader(handle):
            counts.setdefault(row["split"], {"crosswalk": 0, "no_crosswalk": 0})
            counts[row["split"]][row["label"]] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--road-threshold", type=float, default=0.46)
    parser.add_argument("--road-dilate", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(build_road_channel_export(args.source_export, args.output, road_threshold=args.road_threshold, road_dilate=args.road_dilate), indent=2))


if __name__ == "__main__":
    main()

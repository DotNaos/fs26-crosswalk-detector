from __future__ import annotations

import csv
import html
import json
import argparse
from pathlib import Path
import shutil

from PIL import Image, ImageDraw

from crosswalk_detector.scan_backend import SceneRequest, TileRequest, crop_tile, fetch_scene_image


def scene_request(dataset_root: Path, scene_id: str) -> SceneRequest:
    scene = json.loads((dataset_root / "scenes" / scene_id / "scene.json").read_text(encoding="utf8"))
    return SceneRequest(
        scene_id=scene_id,
        latitude=float(scene["latitude"]),
        longitude=float(scene["longitude"]),
        size_m=int(scene["size_m"]),
        image_px=int(scene["image_px"]),
        tile_size_m=int(scene["tile_size_m"]),
    )


def selected_crosswalk_rows(dataset_root: Path, limit: int, min_confidence: float) -> list[dict]:
    index = json.loads((dataset_root / "dataset.json").read_text(encoding="utf8"))
    rows = []
    for shard in index["shards"]:
        for line in (dataset_root / shard["path"]).read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            resolved = row.get("resolved_label", {})
            confidence = resolved.get("confidence")
            if (
                resolved.get("decision") == "crosswalk"
                and isinstance(confidence, int | float)
                and confidence >= min_confidence
            ):
                rows.append(row)
    rows.sort(key=lambda row: float(row.get("resolved_label", {}).get("confidence") or 0.0), reverse=True)
    return rows[:limit]


def write_gallery(output_root: Path, rows: list[dict], dataset_id: str, min_confidence: float) -> None:
    cards = []
    with (output_root / "labels.csv").open(encoding="utf8") as handle:
        for item in csv.DictReader(handle):
            label = item["label"]
            class_name = "crosswalk" if label == "crosswalk" else "not-crosswalk"
            cards.append(
                f"""
      <figure class="{class_name}">
        <img src=\"{html.escape(item["context_path"])}\" loading=\"lazy\" alt=\"{html.escape(item["tile_id"])}\">
        <figcaption><b>#{item["rank"]}</b> {html.escape(item["city"])} · {html.escape(label)} · {html.escape(item["confidence"])}<br><span>{html.escape(item["tile_id"])}</span></figcaption>
      </figure>
    """
            )

    (output_root / "index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAM3 Crosswalk Review</title>
<style>
body {{ margin: 0; font-family: system-ui, sans-serif; background: #f6f7f9; color: #15171a; }}
header {{ position: sticky; top: 0; z-index: 1; background: rgba(246,247,249,.94); backdrop-filter: blur(10px); padding: 14px 18px; border-bottom: 1px solid #d8dde5; }}
h1 {{ font-size: 18px; margin: 0 0 4px; }}
p {{ margin: 0; color: #526070; font-size: 13px; }}
main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; padding: 14px; }}
figure {{ margin: 0; background: white; border: 1px solid #d8dde5; border-radius: 6px; overflow: hidden; }}
figure.crosswalk {{ background: #f0fdf4; border-color: #22c55e; box-shadow: inset 0 0 0 1px rgba(34, 197, 94, .35); }}
figure.not-crosswalk {{ background: #fef2f2; border-color: #ef4444; box-shadow: inset 0 0 0 1px rgba(239, 68, 68, .35); }}
img {{ display: block; width: 100%; aspect-ratio: 1 / 1; object-fit: cover; }}
figcaption {{ padding: 8px; font-size: 12px; line-height: 1.35; }}
figcaption span {{ color: #667386; overflow-wrap: anywhere; }}
</style>
</head>
<body>
<header>
<h1>SAM3 predicted crosswalks: top {len(rows)}</h1>
<p>Generated from {html.escape(dataset_id)}. Sorted by score, showing predictions with score at least {min_confidence}. Green outline marks the selected tile; blue overlay marks the SAM3 pseudo-mask when available. These are not human-confirmed labels.</p>
</header>
<main>
{"".join(cards)}
</main>
</body>
</html>
""",
        encoding="utf8",
    )


def save_context_image(scene_image, scene: SceneRequest, tile: TileRequest, path: Path, mask_path: Path | None = None) -> None:
    context = crop_tile(scene_image, scene, tile, padding_tiles=1.0)
    if mask_path is not None and mask_path.exists():
        _overlay_tile_mask(context, mask_path)
    draw = ImageDraw.Draw(context)
    width, height = context.size
    left = width / 3
    top = height / 3
    right = width * 2 / 3
    bottom = height * 2 / 3
    for inset in range(4):
        draw.rectangle((left + inset, top + inset, right - inset, bottom - inset), outline=(34, 197, 94))
    context.save(path, quality=95)


def _overlay_tile_mask(context: Image.Image, mask_path: Path) -> None:
    width, height = context.size
    left = round(width / 3)
    top = round(height / 3)
    right = round(width * 2 / 3)
    bottom = round(height * 2 / 3)
    mask = Image.open(mask_path).convert("L").resize((right - left, bottom - top), Image.Resampling.NEAREST)
    overlay = Image.new("RGBA", (right - left, bottom - top), (14, 165, 233, 92))
    context.paste(overlay.convert("RGB"), (left, top), mask.point(lambda value: min(180, value)))


def row_mask_path(row: dict) -> Path | None:
    resolved_source = row.get("resolved_label", {}).get("source_id")
    candidates = []
    for label in row.get("labels", []):
        metadata = label.get("metadata") if isinstance(label, dict) else None
        if not isinstance(metadata, dict):
            continue
        artifact = metadata.get("mask_artifact")
        if isinstance(artifact, dict) and artifact.get("path"):
            candidates.append((label.get("source", {}).get("source_id"), Path(str(artifact["path"]))))
    for source_id, path in candidates:
        if source_id == resolved_source:
            return path
    return candidates[-1][1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/sam3-100k-v1"))
    parser.add_argument("--output", type=Path, default=Path("review-exports/sam3-crosswalk-1000"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--min-confidence", type=float, default=0.4)
    args = parser.parse_args()

    dataset_root = args.dataset
    output_root = args.output
    dataset_id = json.loads((dataset_root / "dataset.json").read_text(encoding="utf8"))["dataset_id"]
    image_root = output_root / "images"
    if image_root.exists():
        shutil.rmtree(image_root)
    image_root.mkdir(parents=True, exist_ok=True)

    rows = selected_crosswalk_rows(dataset_root, args.limit, args.min_confidence)
    scene_cache = {}
    csv_path = output_root / "labels.csv"

    with csv_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "image_id",
                "tile_id",
                "scene_id",
                "city",
                "split",
                "label",
                "confidence",
                "image_path",
                "context_path",
                "mask_path",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            scene_id = row["scene_id"]
            scene = scene_request(dataset_root, scene_id)
            if scene_id not in scene_cache:
                print(f"download_scene {scene_id}", flush=True)
                scene_cache[scene_id] = fetch_scene_image(scene)
            tile = TileRequest(
                tile_id=row["tile_id"],
                row=int(row["row"]),
                col=int(row["col"]),
                bbox_mercator=tuple(float(value) for value in row["bbox_mercator"]),
                relative_path=row["reconstruction"]["relative_path"],
            )
            confidence = row.get("resolved_label", {}).get("confidence")
            name = f"{rank:04d}_{row['tile_id'].replace(':', '_')}.jpg"
            image_path = image_root / name
            context_name = f"{rank:04d}_{row['tile_id'].replace(':', '_')}_context.jpg"
            context_path = image_root / context_name
            if not image_path.exists():
                crop_tile(scene_cache[scene_id], scene, tile).save(image_path, quality=95)
            if not context_path.exists():
                save_context_image(scene_cache[scene_id], scene, tile, context_path, row_mask_path(row))
            writer.writerow(
                {
                    "rank": rank,
                    "image_id": row["image_id"],
                    "tile_id": row["tile_id"],
                    "scene_id": scene_id,
                    "city": row["city"],
                    "split": row["split"],
                    "label": row["resolved_label"]["decision"],
                    "confidence": confidence,
                    "image_path": f"images/{name}",
                    "context_path": f"images/{context_name}",
                    "mask_path": row_mask_path(row).as_posix() if row_mask_path(row) is not None else "",
                }
            )

    write_gallery(output_root, rows, dataset_id, args.min_confidence)
    summary = {
        "dataset_id": dataset_id,
        "rows": len(rows),
        "min_confidence": args.min_confidence,
        "sort": "confidence_desc",
        "scenes_downloaded": len(scene_cache),
        "output": str(output_root),
        "labels_csv": str(csv_path),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

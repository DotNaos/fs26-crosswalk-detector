#!/usr/bin/env python3
"""Run CrossMaskNet over a metadata dataset and save static mask predictions."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosswalk_detector.metadata_dataset import resolve_shard_path  # noqa: E402
from crosswalk_detector.scan_backend import SceneRequest, TileRequest, crop_tile, fetch_scene_image  # noqa: E402
from crosswalk_detector.train_crossmask import CrossMaskNet  # noqa: E402
from crosswalk_detector.urban_vision import build_urban_road_density_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Metadata dataset root containing dataset.json.")
    parser.add_argument("--model-root", type=Path, required=True, help="CrossMaskNet model directory.")
    parser.add_argument("--output", type=Path, required=True, help="Prediction output directory.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.005, help="Mask coverage threshold for image-level crosswalk labels.")
    parser.add_argument("--source-id", default="crossmasknet-v4-full")
    parser.add_argument("--display-name", default="CrossMaskNet v4")
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument("--limit-tiles-per-shard", type=int, default=0)
    parser.add_argument("--shard-offset", type=int, default=0, help="Process every Nth shard starting at this zero-based offset.")
    parser.add_argument("--shard-stride", type=int, default=1, help="Shard worker stride for parallel full-dataset runs.")
    parser.add_argument("--scene-cache", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.shard_stride < 1:
        raise SystemExit("--shard-stride must be at least 1.")
    if args.shard_offset < 0 or args.shard_offset >= args.shard_stride:
        raise SystemExit("--shard-offset must be in the range [0, --shard-stride).")
    index = read_json(args.dataset / "dataset.json")
    metrics = read_json(args.model_root / "metrics.json")
    device = choose_device()
    model = load_model(metrics, args.model_root, device)
    image_size = int(metrics["image_size"])
    road_channel = bool(metrics.get("road_channel"))

    if args.output.exists() and args.overwrite:
        import shutil

        shutil.rmtree(args.output)
    (args.output / "shards").mkdir(parents=True, exist_ok=True)
    (args.output / "masks").mkdir(parents=True, exist_ok=True)

    scene_cache: OrderedDict[str, Image.Image] = OrderedDict()
    shards = index["shards"][args.shard_offset :: args.shard_stride]
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    totals = {"tiles": 0, "crosswalk": 0, "no_crosswalk": 0, "masks": 0, "shards": 0}

    for shard_number, shard in enumerate(shards, start=1):
        shard_id = str(shard["shard_id"])
        target = args.output / "shards" / f"{safe_name(shard_id)}.jsonl"
        if args.resume and target.exists():
            print(json.dumps({"event": "skip", "shard": shard_id, "index": shard_number}), flush=True)
            continue
        rows = read_jsonl(resolve_shard_path(args.dataset, shard["path"]))
        if args.limit_tiles_per_shard:
            rows = rows[: args.limit_tiles_per_shard]
        scene = scene_request(args.dataset, str(shard["scene_id"]))
        scene_image = cached_scene_image(scene_cache, scene, max_items=args.scene_cache)
        predictions = run_rows(
            rows,
            scene,
            scene_image,
            model,
            device,
            image_size,
            road_channel=road_channel,
            batch_size=args.batch_size,
            threshold=args.threshold,
            output_root=args.output,
            source_id=args.source_id,
            display_name=args.display_name,
        )
        write_jsonl(target, predictions)
        shard_counts = count_predictions(predictions)
        for key, value in shard_counts.items():
            totals[key] += value
        totals["shards"] += 1
        print(
            json.dumps({"event": "shard", "index": shard_number, "total_shards": len(shards), "shard": shard_id, **shard_counts}),
            flush=True,
        )

    summary = {
        "dataset_id": index["dataset_id"],
        "model_root": str(args.model_root),
        "source_id": args.source_id,
        "display_name": args.display_name,
        "threshold": args.threshold,
        "device": device,
        "created_at": timestamp(),
        "shard_offset": args.shard_offset,
        "shard_stride": args.shard_stride,
        **totals,
    }
    summary_name = "summary.json" if args.shard_stride == 1 else f"summary-worker-{args.shard_offset}-of-{args.shard_stride}.json"
    (args.output / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", **summary}, indent=2), flush=True)
    return 0


def run_rows(
    rows: list[dict[str, Any]],
    scene: SceneRequest,
    scene_image: Image.Image,
    model: CrossMaskNet,
    device: str,
    image_size: int,
    *,
    road_channel: bool,
    batch_size: int,
    threshold: float,
    output_root: Path,
    source_id: str,
    display_name: str,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        tensors = []
        for row in batch_rows:
            tile = tile_request(row)
            crop = crop_tile(scene_image, scene, tile)
            image = crop.resize((image_size, image_size), Image.Resampling.BICUBIC)
            tensor = transform(image)
            if road_channel:
                road_mask = build_urban_road_density_mask(image, max_width=image_size, threshold=0.46).mask
                road_tensor = torch.from_numpy(road_mask.astype("float32")).unsqueeze(0)
                tensor = torch.cat([tensor, road_tensor], dim=0)
            tensors.append(tensor)

        with torch.no_grad():
            logits = model(torch.stack(tensors).to(device))
            probabilities = torch.sigmoid(logits)[:, 0].detach().cpu().numpy()

        for row, probability in zip(batch_rows, probabilities):
            predictions.append(
                prediction_row(
                    row,
                    probability,
                    threshold=threshold,
                    output_root=output_root,
                    source_id=source_id,
                    display_name=display_name,
                )
            )
    return predictions


def prediction_row(
    row: dict[str, Any],
    probability: np.ndarray,
    *,
    threshold: float,
    output_root: Path,
    source_id: str,
    display_name: str,
) -> dict[str, Any]:
    mask = probability >= 0.5
    mask_coverage = float(mask.mean())
    mask_score = float(probability.mean())
    decision = "crosswalk" if mask_coverage >= threshold else "no_crosswalk"
    artifact = save_mask(row, mask, output_root, source_id) if mask.any() else None
    result: dict[str, Any] = {
        "tile_id": str(row["tile_id"]),
        "source_id": source_id,
        "display_name": display_name,
        "decision": decision,
        "confidence": round(max(mask_score, mask_coverage), 6),
        "mask_coverage": round(mask_coverage, 6),
        "mask_score": round(mask_score, 6),
        "model": display_name,
        "run_id": source_id,
        "created_at": timestamp(),
    }
    if artifact:
        result["mask_artifact"] = artifact
    return result


def save_mask(row: dict[str, Any], mask: np.ndarray, output_root: Path, source_id: str) -> dict[str, Any]:
    scene_id = safe_name(str(row["scene_id"]))
    tile_id = safe_name(str(row["tile_id"]))
    relative = Path("masks") / safe_name(source_id) / scene_id / f"{tile_id}.png"
    target = output_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(target, optimize=True)
    return {
        "kind": "crossmasknet-v4-predicted-mask",
        "format": "png",
        "path": target.as_posix(),
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
    }


def cached_scene_image(cache: OrderedDict[str, Image.Image], scene: SceneRequest, *, max_items: int) -> Image.Image:
    if scene.scene_id in cache:
        cache.move_to_end(scene.scene_id)
        return cache[scene.scene_id]
    image = fetch_scene_image(scene)
    cache[scene.scene_id] = image
    while len(cache) > max(1, max_items):
        cache.popitem(last=False)
    return image


def load_model(metrics: dict[str, Any], model_root: Path, device: str) -> CrossMaskNet:
    model = CrossMaskNet(base_channels=int(metrics["base_channels"]), input_channels=int(metrics.get("input_channels", 3))).to(device)
    model.load_state_dict(torch.load(model_root / "crossmasknet_best.pt", map_location=device))
    model.eval()
    return model


def scene_request(dataset_root: Path, scene_id: str) -> SceneRequest:
    scene = read_json(dataset_root / "scenes" / scene_id / "scene.json")
    return SceneRequest(
        scene_id=scene_id,
        latitude=float(scene["latitude"]),
        longitude=float(scene["longitude"]),
        size_m=int(scene["size_m"]),
        image_px=int(scene["image_px"]),
        tile_size_m=int(scene["tile_size_m"]),
    )


def tile_request(row: dict[str, Any]) -> TileRequest:
    return TileRequest(
        tile_id=str(row["tile_id"]),
        row=int(row["row"]),
        col=int(row["col"]),
        bbox_mercator=tuple(float(value) for value in row["bbox_mercator"]),
        relative_path=str(row.get("reconstruction", {}).get("relative_path", "")),
    )


def count_predictions(predictions: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "tiles": len(predictions),
        "crosswalk": sum(1 for row in predictions if row["decision"] == "crosswalk"),
        "no_crosswalk": sum(1 for row in predictions if row["decision"] == "no_crosswalk"),
        "masks": sum(1 for row in predictions if row.get("mask_artifact")),
    }


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")))
            handle.write("\n")


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in value)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())

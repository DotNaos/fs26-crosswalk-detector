from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from crosswalk_detector.train_crossmask import CrossMaskDataset, CrossMaskNet, ExportSample


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _load_rows(export_root: Path) -> list[dict[str, str]]:
    with (export_root / "manifest.csv").open(encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _export_sample(export_root: Path, row: dict[str, str]) -> ExportSample:
    return ExportSample(export_root / row["image_path"], export_root / row["mask_path"], row["label"], row["split"], row["tile_id"])


def _load_model(model_root: Path, device: str) -> tuple[CrossMaskNet, dict]:
    metrics = json.loads((model_root / "metrics.json").read_text(encoding="utf8"))
    model = CrossMaskNet(base_channels=int(metrics["base_channels"]), input_channels=int(metrics.get("input_channels", 3))).to(device)
    model.load_state_dict(torch.load(model_root / "crossmasknet_best.pt", map_location=device))
    model.eval()
    return model, metrics


def _score_negative_rows(export_root: Path, model_root: Path, batch_size: int, num_workers: int) -> list[tuple[float, dict[str, str]]]:
    device = _device()
    model, metrics = _load_model(model_root, device)
    rows = [row for row in _load_rows(export_root) if row["split"] == "train" and row["label"] == "no_crosswalk"]
    samples = [_export_sample(export_root, row) for row in rows]
    loader = DataLoader(
        CrossMaskDataset(samples, image_size=int(metrics["image_size"]), train=False, road_channel=bool(metrics.get("road_channel", False))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    scored: list[tuple[float, dict[str, str]]] = []
    offset = 0
    with torch.no_grad():
        for images, _masks, _tile_ids, _labels in loader:
            probs = torch.sigmoid(model(images.to(device))).detach().cpu().numpy()
            coverages = (probs >= 0.5).mean(axis=(1, 2, 3))
            mean_scores = probs.mean(axis=(1, 2, 3))
            for coverage, mean_score in zip(coverages, mean_scores):
                row = rows[offset]
                score = float(coverage * 10.0 + mean_score)
                scored.append((score, row))
                offset += 1
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def build_hard_export(source_export: Path, model_root: Path, output_root: Path, hard_count: int, repeats: int, batch_size: int, num_workers: int) -> dict:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    rows = _load_rows(source_export)
    hard_rows = [row for _score, row in _score_negative_rows(source_export, model_root, batch_size, num_workers)[:hard_count]]
    fieldnames = list(rows[0].keys()) + ["sample_role"] if "sample_role" not in rows[0] else list(rows[0].keys())
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            copied = {**row, "sample_role": "base"}
            writer.writerow(copied)
        for repeat in range(repeats):
            for row in hard_rows:
                copied = {**row, "sample_role": f"hard_negative_repeat_{repeat + 1}"}
                writer.writerow(copied)
    for folder in ("images", "masks"):
        target = output_root / folder
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source_export / folder, target, symlinks=True)
    source_summary = json.loads((source_export / "summary.json").read_text(encoding="utf8"))
    counts = _count_manifest(output_root / "manifest.csv")
    summary = {
        "source_export": str(source_export),
        "source_model": str(model_root),
        "hard_count": hard_count,
        "repeats": repeats,
        "source_samples": len(rows),
        "hard_added_rows": len(hard_rows) * repeats,
        "samples": len(rows) + len(hard_rows) * repeats,
        "source_summary": source_summary,
        "counts": counts,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    return summary


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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-count", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(build_hard_export(args.source_export, args.model, args.output, args.hard_count, args.repeats, args.batch_size, args.num_workers), indent=2))


if __name__ == "__main__":
    main()

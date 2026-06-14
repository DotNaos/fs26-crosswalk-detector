"""Train a small from-scratch crosswalk segmentation baseline."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .metadata_dataset import resolve_shard_path
from .scan_backend import SceneRequest, TileRequest, crop_tile, fetch_scene_image


@dataclass(frozen=True)
class MaskCandidate:
    image_id: str
    tile_id: str
    scene_id: str
    city: str
    row: int
    col: int
    bbox_mercator: tuple[float, float, float, float]
    relative_path: str
    label: str
    confidence: float
    mask_path: str
    mask_coverage: float


@dataclass(frozen=True)
class ExportSample:
    image_path: Path
    mask_path: Path
    label: str
    split: str
    tile_id: str
    road_path: Path | None = None


class CrossMaskNet(nn.Module):
    """Compact U-Net style model designed for this project, trained from scratch."""

    def __init__(self, base_channels: int = 24, input_channels: int = 3) -> None:
        super().__init__()
        c = base_channels
        self.input_channels = input_channels
        self.enc1 = _conv_block(input_channels, c)
        self.enc2 = _conv_block(c, c * 2)
        self.enc3 = _conv_block(c * 2, c * 4)
        self.enc4 = _conv_block(c * 4, c * 8)
        self.bridge = _conv_block(c * 8, c * 12)
        self.up4 = _decoder_block(c * 12 + c * 8, c * 8)
        self.up3 = _decoder_block(c * 8 + c * 4, c * 4)
        self.up2 = _decoder_block(c * 4 + c * 2, c * 2)
        self.up1 = _decoder_block(c * 2 + c, c)
        self.out = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(images)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        bridge = self.bridge(F.max_pool2d(e4, 2))
        d4 = self.up4(torch.cat([_upsample_like(bridge, e4), e4], dim=1))
        d3 = self.up3(torch.cat([_upsample_like(d4, e3), e3], dim=1))
        d2 = self.up2(torch.cat([_upsample_like(d3, e2), e2], dim=1))
        d1 = self.up1(torch.cat([_upsample_like(d2, e1), e1], dim=1))
        return self.out(d1)


class CrossMaskDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, int]]):
    def __init__(self, rows: list[ExportSample], image_size: int, train: bool, road_channel: bool = False) -> None:
        self.rows = rows
        self.road_channel = road_channel
        self.image_size = image_size
        jitter: list[object] = []
        if train:
            jitter = [
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            ]
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                *jitter,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, int]:
        row = self.rows[index]
        image = Image.open(row.image_path).convert("RGB")
        mask = Image.open(row.mask_path).convert("L")
        image_tensor = self.image_transform(image)
        if self.road_channel:
            if row.road_path is None:
                raise ValueError(f"Missing road_path for road-channel sample: {row.tile_id}")
            road = Image.open(row.road_path).convert("L")
            road_tensor = transforms.functional.resize(
                transforms.functional.to_tensor(road),
                [self.image_size, self.image_size],
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            image_tensor = torch.cat([image_tensor, (road_tensor > 0).float()], dim=0)
        return image_tensor, (self.mask_transform(mask) > 0).float(), row.tile_id, int(row.label == "crosswalk")


def prepare_crossmask_export(
    dataset_root: Path,
    output_root: Path,
    *,
    positive_limit: int = 2500,
    negative_ratio: float = 1.0,
    min_confidence: float = 0.4,
    min_mask_coverage: float = 0.01,
    image_size: int = 128,
    seed: int = 7,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    positives, negatives = _load_candidates(dataset_root, min_confidence=min_confidence, min_mask_coverage=min_mask_coverage)
    positives = sorted(positives, key=lambda item: item.confidence, reverse=True)[:positive_limit]
    rng = random.Random(seed)
    positive_scenes = {row.scene_id for row in positives}
    scene_negatives = [row for row in negatives if row.scene_id in positive_scenes]
    if len(scene_negatives) < round(len(positives) * negative_ratio):
        scene_negatives = negatives
    negative_count = min(len(scene_negatives), round(len(positives) * negative_ratio))
    selected_negatives = rng.sample(scene_negatives, negative_count)
    candidates = positives + selected_negatives
    rng.shuffle(candidates)

    scene_cache: dict[str, Image.Image] = {}
    csv_path = output_root / "manifest.csv"
    image_root = output_root / "images"
    mask_root = output_root / "masks"
    with csv_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "tile_id",
                "scene_id",
                "city",
                "split",
                "label",
                "confidence",
                "mask_coverage",
                "image_path",
                "mask_path",
            ],
        )
        writer.writeheader()
        for item in candidates:
            split = _split_for_id(item.tile_id)
            safe_id = item.image_id.replace(":", "__")
            image_path = image_root / split / item.label / f"{safe_id}.jpg"
            mask_path = mask_root / split / item.label / f"{safe_id}.png"
            if not image_path.exists() or not mask_path.exists():
                scene = _scene_request(dataset_root, item.scene_id)
                scene_image = scene_cache.setdefault(item.scene_id, fetch_scene_image(scene))
                tile = TileRequest(item.tile_id, item.row, item.col, item.bbox_mercator, item.relative_path)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                crop_tile(scene_image, scene, tile).resize((image_size, image_size), Image.Resampling.BICUBIC).save(image_path, quality=94)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                if item.label == "crosswalk":
                    source_mask = Image.open(item.mask_path).convert("L")
                    source_mask.resize((image_size, image_size), Image.Resampling.NEAREST).save(mask_path)
                else:
                    Image.new("L", (image_size, image_size), 0).save(mask_path)
            writer.writerow(
                {
                    "image_id": item.image_id,
                    "tile_id": item.tile_id,
                    "scene_id": item.scene_id,
                    "city": item.city,
                    "split": split,
                    "label": item.label,
                    "confidence": f"{item.confidence:.6f}",
                    "mask_coverage": f"{item.mask_coverage:.6f}",
                    "image_path": image_path.relative_to(output_root).as_posix(),
                    "mask_path": mask_path.relative_to(output_root).as_posix(),
                }
            )

    rows = _load_export_rows(output_root)
    counts = _count_rows(rows)
    summary = {
        "dataset_id": _read_json(dataset_root / "dataset.json")["dataset_id"],
        "positive_limit": positive_limit,
        "negative_ratio": negative_ratio,
        "min_confidence": min_confidence,
        "min_mask_coverage": min_mask_coverage,
        "image_size": image_size,
        "seed": seed,
        "samples": len(rows),
        "counts": counts,
        "scenes_downloaded": len(scene_cache),
        "manifest": str(csv_path),
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def train_crossmask(
    export_root: Path,
    model_root: Path,
    *,
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    image_size: int = 128,
    base_channels: int = 24,
    input_channels: int = 3,
    road_channel: bool = False,
    num_workers: int = 2,
    seed: int = 7,
) -> dict[str, Any]:
    if input_channels != (4 if road_channel else 3):
        raise ValueError("input_channels must be 4 with --road-channel and 3 without it.")
    torch.manual_seed(seed)
    rows = _load_export_rows(export_root)
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val"]
    test_rows = [row for row in rows if row.split == "test"]
    device = _device()
    model_root.mkdir(parents=True, exist_ok=True)
    train_loader = _loader(train_rows, image_size, True, batch_size, num_workers, road_channel)
    val_loader = _loader(val_rows, image_size, False, batch_size, num_workers, road_channel)
    test_loader = _loader(test_rows, image_size, False, batch_size, num_workers, road_channel)

    model = CrossMaskNet(base_channels=base_channels, input_channels=input_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    history: list[dict[str, Any]] = []
    best_state = None
    best_score = -1.0
    pos_weight = _positive_pixel_weight(train_rows)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for images, masks, _tile_ids, _labels in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = _segmentation_loss(logits, masks, pos_weight.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)
        scheduler.step()
        val_metrics = evaluate_crossmask(model, val_loader, device)
        epoch_row = {"epoch": epoch, "train_loss": round(total_loss / max(1, seen), 6), **_round_metrics(val_metrics)}
        history.append(epoch_row)
        if val_metrics["positive_dice"] > best_score:
            best_score = float(val_metrics["positive_dice"])
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
        print(json.dumps(epoch_row), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_crossmask(model, test_loader, device, collect_failures=True)
    model_path = model_root / "crossmasknet_best.pt"
    torch.save(model.state_dict(), model_path)
    metrics = {
        "model": "CrossMaskNet",
        "pretrained": False,
        "export_root": str(export_root),
        "model_path": str(model_path),
        "device": device,
        "epochs": epochs,
        "image_size": image_size,
        "base_channels": base_channels,
        "input_channels": input_channels,
        "road_channel": road_channel,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "counts": _count_rows(rows),
        "history": history,
        "test": _round_metrics(test_metrics),
    }
    _write_json(model_root / "metrics.json", metrics)
    return metrics


def evaluate_crossmask(model: nn.Module, loader: DataLoader, device: str, collect_failures: bool = False) -> dict[str, Any]:
    model.eval()
    total = image_correct = image_tp = image_fp = image_fn = 0
    intersection = union = dice_num = dice_den = 0.0
    positive_intersection = positive_union = positive_dice_num = positive_dice_den = 0.0
    failures: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, masks, tile_ids, labels in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = probs >= 0.5
            mask_bool = masks >= 0.5
            batch_intersection = (preds & mask_bool).float().sum(dim=(1, 2, 3))
            batch_union = (preds | mask_bool).float().sum(dim=(1, 2, 3))
            batch_dice_den = preds.float().sum(dim=(1, 2, 3)) + mask_bool.float().sum(dim=(1, 2, 3))
            intersection += float(batch_intersection.sum().item())
            union += float(batch_union.sum().item())
            dice_num += float((2 * batch_intersection).sum().item())
            dice_den += float(batch_dice_den.sum().item())
            pred_labels = preds.float().mean(dim=(1, 2, 3)) >= 0.005
            true_labels = labels.bool().to(device)
            image_correct += int((pred_labels == true_labels).sum().item())
            image_tp += int((pred_labels & true_labels).sum().item())
            image_fp += int((pred_labels & ~true_labels).sum().item())
            image_fn += int((~pred_labels & true_labels).sum().item())
            total += int(images.size(0))
            positive_indices = torch.where(true_labels)[0]
            if len(positive_indices):
                positive_intersection += float(batch_intersection[positive_indices].sum().item())
                positive_union += float(batch_union[positive_indices].sum().item())
                positive_dice_num += float((2 * batch_intersection[positive_indices]).sum().item())
                positive_dice_den += float(batch_dice_den[positive_indices].sum().item())
            if collect_failures:
                scores = probs.float().mean(dim=(1, 2, 3)).detach().cpu().tolist()
                for tile_id, score, pred_label, true_label in zip(tile_ids, scores, pred_labels.detach().cpu().tolist(), labels.tolist()):
                    if bool(pred_label) != bool(true_label):
                        failures.append({"tile_id": tile_id, "score": round(float(score), 6), "predicted": int(bool(pred_label)), "label": int(true_label)})
    return {
        "pixel_iou": intersection / max(1.0, union),
        "dice": dice_num / max(1.0, dice_den),
        "positive_iou": positive_intersection / max(1.0, positive_union),
        "positive_dice": positive_dice_num / max(1.0, positive_dice_den),
        "image_accuracy": image_correct / max(1, total),
        "image_precision": image_tp / max(1, image_tp + image_fp),
        "image_recall": image_tp / max(1, image_tp + image_fn),
        "samples": total,
        "failures": failures[:100],
    }


def _load_candidates(dataset_root: Path, *, min_confidence: float, min_mask_coverage: float) -> tuple[list[MaskCandidate], list[MaskCandidate]]:
    index = _read_json(dataset_root / "dataset.json")
    positives: list[MaskCandidate] = []
    negatives: list[MaskCandidate] = []
    for shard in index["shards"]:
        for row in _read_jsonl(resolve_shard_path(dataset_root, shard["path"])):
            resolved = row.get("resolved_label", {})
            label = str(resolved.get("decision", ""))
            confidence = float(resolved.get("confidence") or 0.0)
            candidate = _candidate_from_row(row, label, confidence)
            if candidate is None:
                continue
            if label == "crosswalk":
                if confidence >= min_confidence and candidate.mask_coverage >= min_mask_coverage:
                    positives.append(candidate)
            elif label == "no_crosswalk":
                negatives.append(candidate)
    return positives, negatives


def _candidate_from_row(row: dict[str, Any], label: str, confidence: float) -> MaskCandidate | None:
    mask_path = ""
    coverage = 0.0
    if label == "crosswalk":
        mask_path = _row_mask_path(row)
        if not mask_path or not Path(mask_path).exists():
            return None
        coverage = _mask_coverage(Path(mask_path))
    return MaskCandidate(
        image_id=str(row["image_id"]),
        tile_id=str(row["tile_id"]),
        scene_id=str(row["scene_id"]),
        city=str(row.get("city", "")),
        row=int(row["row"]),
        col=int(row["col"]),
        bbox_mercator=tuple(float(value) for value in row["bbox_mercator"]),
        relative_path=str(row["reconstruction"]["relative_path"]),
        label=label,
        confidence=confidence,
        mask_path=mask_path,
        mask_coverage=coverage,
    )


def _row_mask_path(row: dict[str, Any]) -> str:
    resolved_source = row.get("resolved_label", {}).get("source_id")
    candidates = []
    for label in row.get("labels", []):
        metadata = label.get("metadata") if isinstance(label, dict) else None
        if not isinstance(metadata, dict):
            continue
        artifact = metadata.get("mask_artifact")
        if isinstance(artifact, dict) and artifact.get("path"):
            candidates.append((label.get("source", {}).get("source_id"), str(artifact["path"])))
    for source_id, path in candidates:
        if source_id == resolved_source:
            return path
    return candidates[-1][1] if candidates else ""


def _mask_coverage(path: Path) -> float:
    mask = Image.open(path).convert("L")
    arr = np.asarray(mask)
    return float((arr > 0).mean())


def _scene_request(dataset_root: Path, scene_id: str) -> SceneRequest:
    scene = _read_json(dataset_root / "scenes" / scene_id / "scene.json")
    return SceneRequest(scene_id=scene_id, latitude=float(scene["latitude"]), longitude=float(scene["longitude"]), size_m=int(scene["size_m"]), image_px=int(scene["image_px"]), tile_size_m=int(scene["tile_size_m"]))


def _load_export_rows(export_root: Path) -> list[ExportSample]:
    rows = []
    with (export_root / "manifest.csv").open(encoding="utf8") as handle:
        for row in csv.DictReader(handle):
            road_path = export_root / row["road_path"] if row.get("road_path") else None
            rows.append(ExportSample(export_root / row["image_path"], export_root / row["mask_path"], row["label"], row["split"], row["tile_id"], road_path))
    return rows


def _loader(rows: list[ExportSample], image_size: int, train: bool, batch_size: int, num_workers: int, road_channel: bool = False) -> DataLoader:
    return DataLoader(
        CrossMaskDataset(rows, image_size=image_size, train=train, road_channel=road_channel),
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


def _decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return _conv_block(in_channels, out_channels)


def _upsample_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)


def _segmentation_loss(logits: torch.Tensor, masks: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, masks, pos_weight=pos_weight)
    probs = torch.sigmoid(logits)
    intersection = (probs * masks).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * intersection + 1) / (denom + 1)).mean()
    return bce + dice


def _positive_pixel_weight(rows: list[ExportSample]) -> torch.Tensor:
    positive = 0
    total = 0
    for row in rows:
        mask = Image.open(row.mask_path).convert("L")
        arr = np.asarray(mask) > 0
        positive += int(arr.sum())
        total += int(arr.size)
    negative = max(1, total - positive)
    weight = min(20.0, negative / max(1, positive))
    return torch.tensor([weight], dtype=torch.float32)


def _split_for_id(tile_id: str) -> str:
    value = int(hashlib.sha1(tile_id.encode("utf8")).hexdigest()[:8], 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "val"
    return "test"


def _count_rows(rows: list[ExportSample]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        counts.setdefault(row.split, {"crosswalk": 0, "no_crosswalk": 0})
        counts[row.split][row.label] += 1
    return counts


def _round_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    rounded = {}
    for key, value in metrics.items():
        if key == "failures":
            rounded[key] = value
        elif isinstance(value, float):
            rounded[key] = round(value, 6)
        else:
            rounded[key] = value
    return rounded


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]

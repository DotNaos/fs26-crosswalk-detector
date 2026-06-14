from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from crosswalk_detector.train_crossmask import CrossMaskNet
from crosswalk_detector.urban_vision import _dilate, build_urban_road_density_mask


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _load_model(model_root: Path, device: str) -> tuple[CrossMaskNet, dict]:
    metrics = json.loads((model_root / "metrics.json").read_text(encoding="utf8"))
    model = CrossMaskNet(base_channels=int(metrics["base_channels"]), input_channels=int(metrics.get("input_channels", 3))).to(device)
    model.load_state_dict(torch.load(model_root / "crossmasknet_best.pt", map_location=device))
    model.eval()
    return model, metrics


def _predict(model: CrossMaskNet, image: Image.Image, road_input: Image.Image | None, image_size: int, device: str) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_tensor = transform(image)
    if model.input_channels == 4:
        if road_input is None:
            raise ValueError("Model expects a road channel but row has no road_path.")
        road_tensor = transforms.functional.resize(
            transforms.functional.to_tensor(road_input.convert("L")),
            [image_size, image_size],
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        image_tensor = torch.cat([image_tensor, (road_tensor > 0).float()], dim=0)
    with torch.no_grad():
        probs = torch.sigmoid(model(image_tensor.unsqueeze(0).to(device)))[0, 0].detach().cpu().numpy()
    return probs >= 0.5


def _road_context(image: Image.Image, image_size: int, threshold: float, dilate: int) -> np.ndarray:
    result = build_urban_road_density_mask(image.resize((image_size, image_size)), max_width=image_size, threshold=threshold)
    mask = result.mask.astype(bool)
    if dilate > 0:
        mask = _dilate(mask, iterations=dilate)
    return mask


def evaluate(export_root: Path, model_root: Path, road_threshold: float, road_dilate: int, split: str) -> dict:
    device = _device()
    model, metrics = _load_model(model_root, device)
    image_size = int(metrics["image_size"])
    rows = []
    with (export_root / "manifest.csv").open(encoding="utf8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]

    total = image_correct = image_tp = image_fp = image_fn = 0
    intersection = union = dice_num = dice_den = 0.0
    positive_intersection = positive_union = positive_dice_num = positive_dice_den = 0.0
    failures = []
    for row in rows:
        image = Image.open(export_root / row["image_path"]).convert("RGB")
        target = np.asarray(Image.open(export_root / row["mask_path"]).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)) > 0
        road_input = Image.open(export_root / row["road_path"]).convert("L") if row.get("road_path") else None
        raw_pred = _predict(model, image, road_input, image_size, device)
        road = _road_context(image, image_size, road_threshold, road_dilate)
        pred = raw_pred & road
        inter = float((pred & target).sum())
        pred_sum = float(pred.sum())
        target_sum = float(target.sum())
        sample_union = float((pred | target).sum())
        intersection += inter
        union += sample_union
        dice_num += 2 * inter
        dice_den += pred_sum + target_sum
        true_label = row["label"] == "crosswalk"
        pred_label = float(pred.mean()) >= 0.005
        image_correct += int(pred_label == true_label)
        image_tp += int(pred_label and true_label)
        image_fp += int(pred_label and not true_label)
        image_fn += int((not pred_label) and true_label)
        total += 1
        if true_label:
            positive_intersection += inter
            positive_union += sample_union
            positive_dice_num += 2 * inter
            positive_dice_den += pred_sum + target_sum
        if pred_label != true_label and len(failures) < 100:
            failures.append({"tile_id": row["tile_id"], "predicted": int(pred_label), "label": int(true_label), "pred_coverage": round(float(pred.mean()), 6), "road_coverage": round(float(road.mean()), 6)})

    return {
        "model_root": str(model_root),
        "export_root": str(export_root),
        "split": split,
        "road_threshold": road_threshold,
        "road_dilate": road_dilate,
        "pixel_iou": intersection / max(1.0, union),
        "dice": dice_num / max(1.0, dice_den),
        "positive_iou": positive_intersection / max(1.0, positive_union),
        "positive_dice": positive_dice_num / max(1.0, positive_dice_den),
        "image_accuracy": image_correct / max(1, total),
        "image_precision": image_tp / max(1, image_tp + image_fp),
        "image_recall": image_tp / max(1, image_tp + image_fn),
        "samples": total,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--road-threshold", type=float, default=0.46)
    parser.add_argument("--road-dilate", type=int, default=4)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate(args.export, args.model, args.road_threshold, args.road_dilate, args.split)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf8")


if __name__ == "__main__":
    main()

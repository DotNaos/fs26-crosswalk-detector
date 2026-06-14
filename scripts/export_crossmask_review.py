from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms

from crosswalk_detector.train_crossmask import CrossMaskNet
from crosswalk_detector.urban_vision import _dilate, build_urban_road_density_mask


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _overlay(base: Image.Image, mask: Image.Image, color: tuple[int, int, int], alpha: int) -> Image.Image:
    canvas = base.convert("RGBA")
    opacity = mask.convert("L").point(lambda value: min(alpha, round(value * alpha / 255)))
    layer = Image.new("RGBA", base.size, (*color, 0))
    layer.putalpha(opacity)
    return Image.alpha_composite(canvas, layer)


def _load_model(metrics: dict, model_root: Path, device: str) -> CrossMaskNet:
    model = CrossMaskNet(base_channels=int(metrics["base_channels"]), input_channels=int(metrics.get("input_channels", 3))).to(device)
    model.load_state_dict(torch.load(model_root / "crossmasknet_best.pt", map_location=device))
    model.eval()
    return model


def _predict_mask(
    model: CrossMaskNet,
    image: Image.Image,
    image_size: int,
    device: str,
    *,
    road_input: Image.Image | None,
    road_filter: bool,
    road_threshold: float,
    road_dilate: int,
) -> Image.Image:
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
    pred = probs >= 0.5
    if road_filter:
        road = build_urban_road_density_mask(image.resize((image_size, image_size)), max_width=image_size, threshold=road_threshold).mask.astype(bool)
        if road_dilate > 0:
            road = _dilate(road, iterations=road_dilate)
        pred &= road
    return Image.fromarray(pred.astype("uint8") * 255).resize(image.size, Image.Resampling.NEAREST)


def _load_rows(export_root: Path) -> list[dict[str, str]]:
    with (export_root / "manifest.csv").open(encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _selected_rows(rows: list[dict[str, str]], metrics: dict, sample_count: int) -> list[tuple[str, dict[str, str]]]:
    by_tile = {row["tile_id"]: row for row in rows}
    failure_ids = [item["tile_id"] for item in metrics.get("test", {}).get("failures", [])]
    failures = [("failure", by_tile[tile_id]) for tile_id in failure_ids if tile_id in by_tile]
    test_rows = [row for row in rows if row["split"] == "test"]
    failure_set = {row["tile_id"] for _kind, row in failures}
    sample_pool = [row for row in test_rows if row["tile_id"] not in failure_set]
    random.Random(7).shuffle(sample_pool)
    return failures[:sample_count] + [("sample", row) for row in sample_pool[:sample_count]]


def export_review(export_root: Path, model_root: Path, output_root: Path, sample_count: int, target_alpha: int, prediction_alpha: int, road_filter: bool, road_threshold: float, road_dilate: int) -> dict:
    if output_root.exists():
        shutil.rmtree(output_root)
    image_root = output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((model_root / "metrics.json").read_text(encoding="utf8"))
    rows = _load_rows(export_root)
    selected = _selected_rows(rows, metrics, sample_count)
    device = _device()
    model = _load_model(metrics, model_root, device)
    cards = []
    for rank, (kind, row) in enumerate(selected, start=1):
        image = Image.open(export_root / row["image_path"]).convert("RGB")
        target = Image.open(export_root / row["mask_path"]).convert("L").resize(image.size, Image.Resampling.NEAREST)
        road_input = Image.open(export_root / row["road_path"]).convert("L") if row.get("road_path") else None
        predicted = _predict_mask(model, image, int(metrics["image_size"]), device, road_input=road_input, road_filter=road_filter, road_threshold=road_threshold, road_dilate=road_dilate)
        canvas = _overlay(image, target, (239, 68, 68), target_alpha)
        canvas = _overlay(canvas.convert("RGB"), predicted, (14, 165, 233), prediction_alpha)
        draw = ImageDraw.Draw(canvas)
        card_class = _card_class(kind, row["label"])
        outline = _outline_for_class(card_class)
        for inset in range(3):
            draw.rectangle((inset, inset, canvas.width - 1 - inset, canvas.height - 1 - inset), outline=outline)
        name = f"{rank:03d}_{kind}_{row['tile_id']}.png".replace(":", "_")
        canvas.convert("RGB").save(image_root / name)
        cards.append(
            f"""
<figure class="{html.escape(card_class)}">
  <img src="images/{html.escape(name)}" loading="lazy">
  <figcaption><b>{rank}. {html.escape(kind)}</b><br>{html.escape(row["label"])} · conf {html.escape(row["confidence"])} · mask {html.escape(row["mask_coverage"])}<br><span>{html.escape(row["tile_id"])}</span></figcaption>
</figure>
"""
        )
    (output_root / "index.html").write_text(_html(cards), encoding="utf8")
    summary = {
        "items": len(cards),
        "model": metrics["model"],
        "model_root": str(model_root),
        "export_root": str(export_root),
        "target_alpha": target_alpha,
        "prediction_alpha": prediction_alpha,
        "road_filter": road_filter,
        "road_threshold": road_threshold,
        "road_dilate": road_dilate,
        "metrics": metrics["test"],
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    return summary


def _card_class(kind: str, label: str) -> str:
    if kind == "failure":
        return "failure"
    return "positive" if label == "crosswalk" else "negative"


def _outline_for_class(card_class: str) -> tuple[int, int, int, int]:
    if card_class == "failure":
        return (239, 68, 68, 255)
    if card_class == "positive":
        return (34, 197, 94, 255)
    return (148, 163, 184, 255)


def _html(cards: list[str]) -> str:
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CrossMask review</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f6f7f9;color:#15171a}}
header{{position:sticky;top:0;background:rgba(246,247,249,.95);backdrop-filter:blur(10px);padding:14px 18px;border-bottom:1px solid #d8dde5;z-index:1}}
h1{{font-size:18px;margin:0 0 4px}}
p{{margin:0;color:#526070;font-size:13px}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;padding:14px}}
figure{{margin:0;background:white;border:1px solid #d8dde5;border-radius:6px;overflow:hidden}}
figure.failure{{border-color:#ef4444;background:#fef2f2}}
figure.positive{{border-color:#22c55e;background:#f0fdf4}}
figure.negative{{border-color:#94a3b8;background:#f8fafc}}
img{{display:block;width:100%;aspect-ratio:1/1;object-fit:cover}}
figcaption{{padding:8px;font-size:12px;line-height:1.35}}
span{{color:#667386;overflow-wrap:anywhere}}
</style>
</head>
<body>
<header>
<h1>CrossMask prediction review</h1>
<p>Transparent red = target mask. Transparent blue = model prediction. Red border = wrong image-level decision. Green border = crosswalk sample. Gray border = correct no-crosswalk sample.</p>
</header>
<main>
{"".join(cards)}
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=48)
    parser.add_argument("--target-alpha", type=int, default=54)
    parser.add_argument("--prediction-alpha", type=int, default=72)
    parser.add_argument("--road-filter", action="store_true")
    parser.add_argument("--road-threshold", type=float, default=0.46)
    parser.add_argument("--road-dilate", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            export_review(
                args.export,
                args.model,
                args.output,
                args.sample_count,
                args.target_alpha,
                args.prediction_alpha,
                args.road_filter,
                args.road_threshold,
                args.road_dilate,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

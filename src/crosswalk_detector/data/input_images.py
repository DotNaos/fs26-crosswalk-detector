"""Prepare plain input image folders for CrossMaskNet prediction runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import re
from typing import Any

from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from .raw_imagery import load_cached_scene_image
from ..scan.scan_backend import TileRequest, crop_tile
from ..models.train_crossmask import MaskCandidate, _load_candidates, _scene_request


def download_input_images(
    dataset_root: Path,
    output_dir: Path,
    *,
    positive_count: int = 10,
    negative_count: int = 10,
    image_size: int = 128,
    seed: int = 7,
    min_confidence: float = 0.4,
    min_mask_coverage: float = 0.01,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in output_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                path.unlink()

    _progress_line("Selecting source images...", enabled=show_progress)
    positives, negatives = _load_candidates(dataset_root, min_confidence=min_confidence, min_mask_coverage=min_mask_coverage)
    selected = _select_candidates(positives, positive_count, "positive", seed)
    selected += _select_candidates(negatives, negative_count, "negative", seed + 1)

    rows: list[dict[str, Any]] = []
    scene_cache: dict[str, Image.Image] = {}
    _progress_line(f"Preparing input images: starting {len(selected)} image(s)", enabled=show_progress)
    progress = _progress() if show_progress else None
    progress_context = progress if progress is not None else _NoProgress()
    with progress_context:
        task = progress.add_task("Preparing input images", total=len(selected)) if progress is not None else None
        for index, candidate in enumerate(selected, start=1):
            scene = _scene_request(dataset_root, candidate.scene_id)
            scene_image = scene_cache.setdefault(candidate.scene_id, load_cached_scene_image(dataset_root, scene))
            tile = TileRequest(candidate.tile_id, candidate.row, candidate.col, candidate.bbox_mercator, candidate.relative_path)
            image = crop_tile(scene_image, scene, tile).resize((image_size, image_size), Image.Resampling.BICUBIC)
            file_name = f"input-{index:04d}-{_safe_name(candidate.tile_id)}.jpg"
            image_path = output_dir / file_name
            image.save(image_path, quality=94)
            rows.append(
                {
                    "image_path": str(image_path),
                    "source_label": _source_label(candidate),
                    "tile_id": candidate.tile_id,
                    "scene_id": candidate.scene_id,
                    "city": candidate.city,
                    "confidence": round(candidate.confidence, 6),
                    "mask_coverage": round(candidate.mask_coverage, 6),
                }
            )
            if progress is not None and task is not None:
                progress.update(task, advance=1)
                progress.refresh()
            if _should_emit_progress(index, len(selected)):
                _progress_line(f"Preparing input images: {index}/{len(selected)}", enabled=show_progress)

    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "image_size": image_size,
        "requested_positive": positive_count,
        "requested_negative": negative_count,
        "positive": sum(1 for row in rows if row["source_label"] == "positive"),
        "negative": sum(1 for row in rows if row["source_label"] == "negative"),
        "total": len(rows),
        "seed": seed,
        "manifest": str(output_dir / "input_images.csv"),
    }
    _write_files(output_dir, rows, summary)
    return {**summary, "images": rows}


def _select_candidates(candidates: list[MaskCandidate], count: int, label: str, seed: int) -> list[MaskCandidate]:
    if count <= 0:
        return []
    if len(candidates) < count:
        raise ValueError(f"Requested {count} {label} images, but only {len(candidates)} are available.")
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:count]


def _source_label(candidate: MaskCandidate) -> str:
    return "positive" if candidate.label == "crosswalk" else "negative"


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return safe[:96] or "image"


def _write_files(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(json.dumps({**summary, "images": rows}, indent=2), encoding="utf8")
    with (output_dir / "input_images.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "source_label", "tile_id", "scene_id", "city", "confidence", "mask_coverage"],
        )
        writer.writeheader()
        writer.writerows(rows)


class _NoProgress:
    def __enter__(self) -> "_NoProgress":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(force_terminal=True),
        refresh_per_second=20,
        redirect_stdout=False,
        redirect_stderr=False,
    )


def _progress_line(message: str, *, enabled: bool) -> None:
    if not enabled:
        return
    console = Console(force_terminal=True)
    console.print(f"[cyan]{message}[/cyan]")
    console.file.flush()


def _should_emit_progress(completed: int, total: int) -> bool:
    if completed <= 1 or completed >= total:
        return True
    return completed % max(1, total // 10) == 0

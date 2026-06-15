"""Metadata-only dataset contract helpers for JSONL crosswalk datasets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_FORMAT = "crosswalk-jsonl-v1"
LABEL_DECISIONS = {"crosswalk", "no_crosswalk", "drop"}
LABEL_SOURCE_KINDS = {"model", "human"}
REVIEW_STATES = {"unreviewed", "reviewed", "disputed", "dropped"}
SPLITS = {"train", "val", "test"}
RESOLUTION_MODES = {"human_override", "priority", "weighted_vote"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


@dataclass(frozen=True)
class ValidationSummary:
    dataset_id: str
    shard_count: int
    row_count: int
    selected_count: int
    errors: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def load_metadata_dataset_index(dataset_root: Path) -> dict[str, Any]:
    index_path = dataset_root / "dataset.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing metadata dataset index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf8"))
    if index.get("format") != SUPPORTED_FORMAT:
        raise ValueError(f"Unsupported metadata dataset format: {index.get('format')}")
    if not isinstance(index.get("shards"), list):
        raise ValueError("Metadata dataset index has no shard list.")
    return index


def validate_metadata_dataset(dataset_root: Path, max_errors: int = 200) -> ValidationSummary:
    errors: list[ValidationIssue] = []
    try:
        index = load_metadata_dataset_index(dataset_root)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return ValidationSummary("", 0, 0, 0, (ValidationIssue(str(dataset_root / "dataset.json"), str(exc)),))

    dataset_id = str(index.get("dataset_id", ""))
    shards = index["shards"]
    seen_image_ids: set[str] = set()
    seen_tile_ids: set[str] = set()
    total_rows = 0
    selected_rows = 0

    _validate_index(index, errors)
    _validate_no_images_in_dataset_root(dataset_root, errors, max_errors)

    for shard_index, shard in enumerate(shards):
        shard_label = f"dataset.json:shards[{shard_index}]"
        shard_path_value = shard.get("path") if isinstance(shard, dict) else None
        if not isinstance(shard_path_value, str):
            _add_error(errors, shard_label, "Shard path must be a string.", max_errors)
            continue
        try:
            shard_path = resolve_shard_path(dataset_root, shard_path_value)
        except ValueError as exc:
            _add_error(errors, shard_label, str(exc), max_errors)
            continue
        if not shard_path.exists():
            _add_error(errors, shard_path_value, "Shard file does not exist.", max_errors)
            continue

        shard_rows = 0
        for line_number, row in _iter_jsonl_rows(shard_path, errors, max_errors):
            row_path = f"{shard_path_value}:{line_number}"
            shard_rows += 1
            total_rows += 1
            if isinstance(row, dict):
                if row.get("selected_for_training") is True:
                    selected_rows += 1
                _validate_image_row(row, row_path, seen_image_ids, seen_tile_ids, errors, max_errors)
            else:
                _add_error(errors, row_path, "JSONL row must be an object.", max_errors)

        expected_shard_rows = shard.get("tile_count") if isinstance(shard, dict) else None
        if isinstance(expected_shard_rows, int) and expected_shard_rows != shard_rows:
            _add_error(
                errors,
                shard_label,
                f"Shard declares {expected_shard_rows} rows but contains {shard_rows}.",
                max_errors,
            )

    if isinstance(index.get("tile_count"), int) and index["tile_count"] != total_rows:
        _add_error(
            errors,
            "dataset.json:tile_count",
            f"Index declares {index['tile_count']} rows but shards contain {total_rows}.",
            max_errors,
        )
    if isinstance(index.get("selected_count"), int) and index["selected_count"] != selected_rows:
        _add_error(
            errors,
            "dataset.json:selected_count",
            f"Index declares {index['selected_count']} selected rows but shards contain {selected_rows}.",
            max_errors,
        )

    return ValidationSummary(dataset_id, len(shards), total_rows, selected_rows, tuple(errors))


def resolve_label_votes(labels: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    votes = [label for label in labels if isinstance(label, dict)]
    valid_votes = [vote for vote in votes if vote.get("decision") in LABEL_DECISIONS and isinstance(vote.get("source"), dict)]
    if not valid_votes:
        return None

    human_votes = [vote for vote in valid_votes if vote["source"].get("kind") == "human"]
    if human_votes:
        winning = _newest_vote(human_votes)
        return _resolved_from_vote(winning, "human_override")

    winning = max(
        valid_votes,
        key=lambda vote: (
            _source_priority(vote),
            _confidence(vote),
            str(vote.get("created_at", "")),
        ),
    )
    return _resolved_from_vote(winning, "priority")


def resolve_shard_path(dataset_root: Path, relative_path: str) -> Path:
    if relative_path.startswith("/") or ".." in Path(relative_path).parts:
        raise ValueError(f"Unsafe metadata shard path: {relative_path}")
    root = dataset_root.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Metadata shard path escapes dataset root: {relative_path}")
    return path


def _validate_index(index: dict[str, Any], errors: list[ValidationIssue]) -> None:
    _require_string(index, "dataset_id", "dataset.json", errors)
    _require_string(index, "run_name", "dataset.json", errors)
    _require_string(index, "export_name", "dataset.json", errors)
    _require_int(index, "tile_count", "dataset.json", errors)
    _require_int(index, "selected_count", "dataset.json", errors)
    _require_int(index, "shard_target_count", "dataset.json", errors)


def _validate_image_row(
    row: dict[str, Any],
    row_path: str,
    seen_image_ids: set[str],
    seen_tile_ids: set[str],
    errors: list[ValidationIssue],
    max_errors: int,
) -> None:
    image_id = _require_string(row, "image_id", row_path, errors, max_errors)
    tile_id = _require_string(row, "tile_id", row_path, errors, max_errors)
    if image_id:
        _validate_unique(image_id, seen_image_ids, row_path, "image_id", errors, max_errors)
    if tile_id:
        _validate_unique(tile_id, seen_tile_ids, row_path, "tile_id", errors, max_errors)

    for field in ("scene_id", "source_scene_id", "perimeter_id", "city"):
        _require_string(row, field, row_path, errors, max_errors)
    _require_choice(row, "split", SPLITS, row_path, errors, max_errors)
    _require_int(row, "row", row_path, errors, max_errors)
    _require_int(row, "col", row_path, errors, max_errors)
    _require_bbox(row.get("bbox_mercator"), f"{row_path}:bbox_mercator", errors, max_errors)
    _require_object(row, "swisstopo", row_path, errors, max_errors)
    _validate_swisstopo(row.get("swisstopo"), row_path, errors, max_errors)
    _require_object(row, "reconstruction", row_path, errors, max_errors)
    _validate_reconstruction(row.get("reconstruction"), row_path, errors, max_errors)
    if "road_overlay_ref" in row and row["road_overlay_ref"] is not None:
        _validate_road_overlay_ref(row["road_overlay_ref"], row_path, errors, max_errors)
    _validate_labels(row.get("labels"), row_path, errors, max_errors)
    _validate_resolved_label(row.get("resolved_label"), row_path, errors, max_errors)
    _require_choice(row, "review_state", REVIEW_STATES, row_path, errors, max_errors)
    if not isinstance(row.get("selected_for_training"), bool):
        _add_error(errors, f"{row_path}:selected_for_training", "Expected a boolean.", max_errors)


def _validate_swisstopo(value: Any, row_path: str, errors: list[ValidationIssue], max_errors: int) -> None:
    if not isinstance(value, dict):
        return
    if value.get("provider") != "swisstopo":
        _add_error(errors, f"{row_path}:swisstopo.provider", "Expected provider to be 'swisstopo'.", max_errors)
    if value.get("product") != "SWISSIMAGE":
        _add_error(errors, f"{row_path}:swisstopo.product", "Expected product to be 'SWISSIMAGE'.", max_errors)
    if value.get("access") not in {"stac-cog", "wmts"}:
        _add_error(errors, f"{row_path}:swisstopo.access", "Expected access to be 'stac-cog' or 'wmts'.", max_errors)
    if value.get("crs") != "EPSG:2056":
        _add_error(errors, f"{row_path}:swisstopo.crs", "Expected CRS to be EPSG:2056.", max_errors)
    if not value.get("asset_id") and not value.get("asset_url"):
        _add_error(errors, f"{row_path}:swisstopo", "Expected asset_id or asset_url.", max_errors)


def _validate_reconstruction(value: Any, row_path: str, errors: list[ValidationIssue], max_errors: int) -> None:
    if not isinstance(value, dict):
        return
    for field in ("source_scene_id", "relative_path"):
        _require_string(value, field, f"{row_path}:reconstruction", errors, max_errors)
    for field in ("row", "col", "tile_size_m"):
        _require_int(value, field, f"{row_path}:reconstruction", errors, max_errors)
    _require_bbox(value.get("tile_bbox_mercator"), f"{row_path}:reconstruction.tile_bbox_mercator", errors, max_errors)
    crop = value.get("crop_px")
    if not isinstance(crop, dict):
        _add_error(errors, f"{row_path}:reconstruction.crop_px", "Expected an object.", max_errors)
        return
    for field in ("left", "top", "width", "height"):
        _require_int(crop, field, f"{row_path}:reconstruction.crop_px", errors, max_errors)


def _validate_road_overlay_ref(value: Any, row_path: str, errors: list[ValidationIssue], max_errors: int) -> None:
    if not isinstance(value, dict):
        _add_error(errors, f"{row_path}:road_overlay_ref", "Expected an object.", max_errors)
        return
    for field in ("overlay_id", "perimeter_id", "cell_id"):
        _require_string(value, field, f"{row_path}:road_overlay_ref", errors, max_errors)
    if not isinstance(value.get("surface_ratio"), int | float):
        _add_error(errors, f"{row_path}:road_overlay_ref.surface_ratio", "Expected a number.", max_errors)


def _validate_labels(value: Any, row_path: str, errors: list[ValidationIssue], max_errors: int) -> None:
    if not isinstance(value, list):
        _add_error(errors, f"{row_path}:labels", "Expected an array.", max_errors)
        return
    for index, label in enumerate(value):
        label_path = f"{row_path}:labels[{index}]"
        if not isinstance(label, dict):
            _add_error(errors, label_path, "Expected an object.", max_errors)
            continue
        _require_string(label, "vote_id", label_path, errors, max_errors)
        _require_choice(label, "decision", LABEL_DECISIONS, label_path, errors, max_errors)
        _require_string(label, "created_at", label_path, errors, max_errors)
        source = label.get("source")
        if not isinstance(source, dict):
            _add_error(errors, f"{label_path}:source", "Expected an object.", max_errors)
            continue
        _require_string(source, "source_id", f"{label_path}:source", errors, max_errors)
        _require_string(source, "display_name", f"{label_path}:source", errors, max_errors)
        _require_choice(source, "kind", LABEL_SOURCE_KINDS, f"{label_path}:source", errors, max_errors)
        _require_int(source, "priority", f"{label_path}:source", errors, max_errors)


def _validate_resolved_label(value: Any, row_path: str, errors: list[ValidationIssue], max_errors: int) -> None:
    if not isinstance(value, dict):
        _add_error(errors, f"{row_path}:resolved_label", "Expected an object.", max_errors)
        return
    _require_choice(value, "decision", LABEL_DECISIONS, f"{row_path}:resolved_label", errors, max_errors)
    _require_string(value, "source_id", f"{row_path}:resolved_label", errors, max_errors)
    _require_choice(value, "source_kind", LABEL_SOURCE_KINDS, f"{row_path}:resolved_label", errors, max_errors)
    _require_choice(value, "resolved_by", RESOLUTION_MODES, f"{row_path}:resolved_label", errors, max_errors)
    _require_string(value, "updated_at", f"{row_path}:resolved_label", errors, max_errors)


def _iter_jsonl_rows(
    path: Path,
    errors: list[ValidationIssue],
    max_errors: int,
) -> Iterable[tuple[int, Any]]:
    with path.open(encoding="utf8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_number, json.loads(stripped)
            except json.JSONDecodeError as exc:
                _add_error(errors, f"{path}:{line_number}", f"Invalid JSON: {exc}", max_errors)


def _validate_no_images_in_dataset_root(dataset_root: Path, errors: list[ValidationIssue], max_errors: int) -> None:
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            _add_error(errors, str(path), "Image file is inside metadata dataset root.", max_errors)


def _require_string(
    value: dict[str, Any],
    field: str,
    path: str,
    errors: list[ValidationIssue],
    max_errors: int = 200,
) -> str | None:
    if not isinstance(value.get(field), str) or not value[field]:
        _add_error(errors, f"{path}:{field}", "Expected a non-empty string.", max_errors)
        return None
    return str(value[field])


def _require_int(
    value: dict[str, Any],
    field: str,
    path: str,
    errors: list[ValidationIssue],
    max_errors: int = 200,
) -> int | None:
    if not isinstance(value.get(field), int):
        _add_error(errors, f"{path}:{field}", "Expected an integer.", max_errors)
        return None
    return int(value[field])


def _require_object(
    value: dict[str, Any],
    field: str,
    path: str,
    errors: list[ValidationIssue],
    max_errors: int,
) -> None:
    if not isinstance(value.get(field), dict):
        _add_error(errors, f"{path}:{field}", "Expected an object.", max_errors)


def _require_choice(
    value: dict[str, Any],
    field: str,
    allowed: set[str],
    path: str,
    errors: list[ValidationIssue],
    max_errors: int = 200,
) -> None:
    if value.get(field) not in allowed:
        _add_error(errors, f"{path}:{field}", f"Expected one of {sorted(allowed)}.", max_errors)


def _require_bbox(value: Any, path: str, errors: list[ValidationIssue], max_errors: int) -> None:
    if not isinstance(value, list) or len(value) != 4 or not all(isinstance(item, int | float) for item in value):
        _add_error(errors, path, "Expected four numeric bbox values.", max_errors)


def _validate_unique(
    value: str,
    seen: set[str],
    path: str,
    field: str,
    errors: list[ValidationIssue],
    max_errors: int,
) -> None:
    if value in seen:
        _add_error(errors, f"{path}:{field}", f"Duplicate {field}: {value}", max_errors)
    seen.add(value)


def _source_priority(vote: dict[str, Any]) -> int:
    priority = vote.get("source", {}).get("priority", 0)
    return priority if isinstance(priority, int) else 0


def _confidence(vote: dict[str, Any]) -> float:
    confidence = vote.get("confidence", 0.0)
    return float(confidence) if isinstance(confidence, int | float) else 0.0


def _newest_vote(votes: list[dict[str, Any]]) -> dict[str, Any]:
    return max(votes, key=lambda vote: str(vote.get("created_at", "")))


def _resolved_from_vote(vote: dict[str, Any], mode: str) -> dict[str, Any]:
    source = vote["source"]
    resolved: dict[str, Any] = {
        "decision": vote["decision"],
        "source_id": source["source_id"],
        "source_kind": source["kind"],
        "resolved_by": mode,
        "updated_at": vote["created_at"],
    }
    if isinstance(vote.get("confidence"), int | float):
        resolved["confidence"] = vote["confidence"]
    return resolved


def _add_error(errors: list[ValidationIssue], path: str, message: str, max_errors: int) -> None:
    if len(errors) < max_errors:
        errors.append(ValidationIssue(path, message))


__all__ = [
    "ValidationIssue",
    "ValidationSummary",
    "load_metadata_dataset_index",
    "resolve_label_votes",
    "resolve_shard_path",
    "validate_metadata_dataset",
]

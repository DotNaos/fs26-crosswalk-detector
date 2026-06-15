"""SAM3 result summary helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


SCORE_THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8)


@dataclass(frozen=True)
class Sam3ResultsSummary:
    result_files: int
    expected_shards: int | None
    remaining_shards: int | None
    tiles: int
    crosswalk: int
    no_crosswalk: int
    masks: int
    score_bins: dict[str, int]
    bad_no_crosswalk_mask_refs: int
    missing_crosswalk_mask_refs: int
    missing_mask_files: int

    @property
    def mask_consistent(self) -> bool:
        return (
            self.masks == self.crosswalk
            and self.bad_no_crosswalk_mask_refs == 0
            and self.missing_crosswalk_mask_refs == 0
            and self.missing_mask_files == 0
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mask_consistent"] = self.mask_consistent
        return payload


def summarize_sam3_results(results_root: Path, *, expected_shards: int | None = None) -> Sam3ResultsSummary:
    files = sorted(results_root.glob("shard-*.result.json"))
    score_bins = {f">={threshold:g}": 0 for threshold in SCORE_THRESHOLDS}
    tiles = crosswalk = no_crosswalk = masks = 0
    bad_no_crosswalk_mask_refs = 0
    missing_crosswalk_mask_refs = 0
    missing_mask_files = 0

    for path in files:
        payload = json.loads(path.read_text(encoding="utf8"))
        summary = payload.get("summary", {})
        tiles += int(summary.get("total", 0))
        crosswalk += int(summary.get("crosswalk", 0))
        no_crosswalk += int(summary.get("no_crosswalk", 0))
        for tile in payload.get("tiles", []):
            if not isinstance(tile, dict):
                continue
            label = tile.get("label")
            artifact = tile.get("mask_artifact")
            if label == "crosswalk":
                _add_score_bins(score_bins, float(tile.get("score") or 0.0))
                if not isinstance(artifact, dict):
                    missing_crosswalk_mask_refs += 1
                elif not _artifact_path(artifact).exists():
                    missing_mask_files += 1
            elif artifact is not None:
                bad_no_crosswalk_mask_refs += 1

    mask_root = results_root / "masks"
    if mask_root.exists():
        masks = sum(1 for _ in mask_root.rglob("*.png"))
    remaining_shards = expected_shards - len(files) if expected_shards is not None else None
    return Sam3ResultsSummary(
        result_files=len(files),
        expected_shards=expected_shards,
        remaining_shards=remaining_shards,
        tiles=tiles,
        crosswalk=crosswalk,
        no_crosswalk=no_crosswalk,
        masks=masks,
        score_bins=score_bins,
        bad_no_crosswalk_mask_refs=bad_no_crosswalk_mask_refs,
        missing_crosswalk_mask_refs=missing_crosswalk_mask_refs,
        missing_mask_files=missing_mask_files,
    )


def _add_score_bins(score_bins: dict[str, int], score: float) -> None:
    for threshold in SCORE_THRESHOLDS:
        if score >= threshold:
            score_bins[f">={threshold:g}"] += 1


def _artifact_path(artifact: dict[str, Any]) -> Path:
    return Path(str(artifact.get("path", "")))

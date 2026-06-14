import numpy as np
from PIL import Image, ImageDraw

from dataclasses import replace

from crosswalk_detector.scan_backend import Sam31CrosswalkScanner, Sam31Detection, Sam31TileMetrics, SceneRequest, TileRequest


def _scanner() -> Sam31CrosswalkScanner:
    scanner = object.__new__(Sam31CrosswalkScanner)
    scanner.tile_threshold = 0.18
    scanner.negative_tile_threshold = 0.18
    scanner.edge_center_threshold = 0.45
    scanner.parking_grid_max_score = 0.60
    return scanner


def _scene() -> SceneRequest:
    return SceneRequest(scene_id="test-scene", latitude=0.0, longitude=0.0, size_m=100, image_px=100, tile_size_m=100)


def _tile() -> TileRequest:
    return TileRequest(tile_id="test-tile", row=0, col=0, bbox_mercator=(-50.0, -50.0, 50.0, 50.0), relative_path="tile.jpg")


def _detection(prompt: str, score: float, box: tuple[float, float, float, float], polarity: str) -> Sam31Detection:
    mask = np.zeros((100, 100), dtype=bool)
    left, top, right, bottom = [int(value) for value in box]
    mask[top:bottom, left:right] = True
    return Sam31Detection(prompt=prompt, score=score, box=box, mask=mask, polarity=polarity)


def test_sam31_positive_detection_scores_crosswalk() -> None:
    metrics = _scanner().score_tile(
        _scene(),
        _tile(),
        [_detection("pedestrian crosswalk", 0.42, (20.0, 20.0, 60.0, 60.0), "positive")],
    )

    assert metrics.label == "crosswalk"
    assert metrics.score == 0.42
    assert metrics.prompt == "pedestrian crosswalk"
    assert metrics.mask is not None
    assert metrics.mask.shape == (100, 100)


def test_sam31_negative_detection_suppresses_matching_crosswalk_candidate() -> None:
    metrics = _scanner().score_tile(
        _scene(),
        _tile(),
        [
            _detection("pedestrian crosswalk", 0.42, (20.0, 20.0, 60.0, 60.0), "positive"),
            _detection("parking markings", 0.31, (21.0, 21.0, 61.0, 61.0), "negative"),
        ],
    )

    assert metrics.label == "no_crosswalk"
    assert metrics.score == 0.0
    assert metrics.prompt == "suppressed:parking markings"


def test_sam31_negative_detection_does_not_suppress_unmatched_candidate() -> None:
    metrics = _scanner().score_tile(
        _scene(),
        _tile(),
        [
            _detection("pedestrian crosswalk", 0.42, (20.0, 20.0, 60.0, 60.0), "positive"),
            _detection("parking markings", 0.31, (70.0, 70.0, 95.0, 95.0), "negative"),
        ],
    )

    assert metrics.label == "crosswalk"
    assert metrics.score == 0.42
    assert metrics.prompt == "pedestrian crosswalk"


def test_sam31_low_score_edge_overlap_is_suppressed() -> None:
    metrics = _scanner().score_tile(
        _scene(),
        _tile(),
        [_detection("pedestrian crosswalk", 0.42, (90.0, 20.0, 130.0, 60.0), "positive")],
    )

    assert metrics.label == "no_crosswalk"
    assert metrics.score == 0.0
    assert metrics.prompt == "suppressed:edge-overlap"
    assert metrics.mask is None


def test_sam31_rejects_low_score_yellow_parking_grid_image() -> None:
    image = Image.new("RGB", (64, 64), (95, 95, 88))
    draw = ImageDraw.Draw(image)
    for x in range(8, 57, 12):
        draw.line((x, 4, x, 60), fill=(190, 175, 60), width=3)
    for y in range(8, 57, 12):
        draw.line((4, y, 60, y), fill=(190, 175, 60), width=3)
    metrics = Sam31TileMetrics(
        label="crosswalk",
        score=0.42,
        peak=0.42,
        coverage=0.3,
        box_overlap=0.3,
        prompt="pedestrian crosswalk",
        detection_count=1,
    )

    assert _scanner().image_rejection_reason(image, metrics) == "yellow-parking-grid"


def test_suppressed_tile_metrics_drop_mask() -> None:
    metrics = Sam31TileMetrics(
        label="crosswalk",
        score=0.42,
        peak=0.42,
        coverage=0.3,
        box_overlap=0.3,
        prompt="pedestrian crosswalk",
        detection_count=1,
        mask=np.ones((8, 8), dtype=bool),
    )

    suppressed = replace(
        metrics,
        label="no_crosswalk",
        score=0.0,
        peak=0.0,
        coverage=0.0,
        box_overlap=0.0,
        prompt="suppressed:yellow-parking-grid",
        mask=None,
    )

    assert suppressed.mask is None

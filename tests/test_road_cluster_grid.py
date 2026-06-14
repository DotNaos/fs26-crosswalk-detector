from crosswalk_detector.road_cluster_grid import _bounded_cell_size, _keep_cluster_cell


def test_zoomed_road_clusters_use_25_meter_cells_when_view_is_small_enough() -> None:
    bbox = (1_060_000.0, 5_917_000.0, 1_061_500.0, 5_918_500.0)

    assert _bounded_cell_size(bbox, 16.5, max_cells=9000) == 25


def test_road_cluster_cells_require_real_road_density() -> None:
    assert _keep_cluster_cell(
        surface_ratio=0.95,
        density_score=0.9,
        road_pixel_ratio=0.02,
        line_density_score=0.08,
        cell_size_m=25,
        surface_threshold=0.55,
    ) is False
    assert _keep_cluster_cell(
        surface_ratio=0.6,
        density_score=0.55,
        road_pixel_ratio=0.22,
        line_density_score=0.44,
        cell_size_m=25,
        surface_threshold=0.55,
    ) is True

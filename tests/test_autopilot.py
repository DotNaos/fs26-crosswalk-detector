from PIL import Image, ImageDraw

from crosswalk_detector.geo.autopilot import GeoReference, WEB_MERCATOR_WORLD_MIN, _aligned_grid_origin, _build_bvh_cells, _ranked_panel_cells, build_urban_grid


def test_build_urban_grid_detects_clusters_without_city_coordinates() -> None:
    image = Image.new("RGB", (160, 100), (48, 92, 42))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 18, 56, 46), fill=(138, 138, 130))
    draw.rectangle((98, 54, 136, 82), fill=(150, 145, 136))
    for offset in range(0, 36, 6):
        draw.line((20 + offset, 18, 56, 46 - offset // 2), fill=(210, 210, 196), width=1)
        draw.line((98, 54 + offset // 2, 136, 82 - offset), fill=(215, 215, 200), width=1)

    plan = build_urban_grid(
        image,
        GeoReference(west=0, south=0, east=16, north=10),
        rows=10,
        cols=16,
        threshold=0.44,
        min_cluster_cells=2,
    )

    assert len(plan.cells) == 160
    assert len(plan.clusters) >= 2
    assert all(cluster.bbox[0] < cluster.bbox[2] for cluster in plan.clusters)
    assert all(cluster.bbox[1] < cluster.bbox[3] for cluster in plan.clusters)


def test_bvh_and_scene_cells_are_square_and_aligned_to_training_grid() -> None:
    image = Image.new("RGB", (160, 100), (48, 92, 42))
    draw = ImageDraw.Draw(image)
    draw.rectangle((36, 28, 124, 76), fill=(148, 148, 138))
    for offset in range(0, 80, 8):
        draw.line((36 + offset, 28, 124 - offset // 3, 76), fill=(215, 215, 200), width=1)

    plan = build_urban_grid(
        image,
        GeoReference(west=5.86, south=45.78, east=10.55, north=47.84),
        rows=10,
        cols=16,
        threshold=0.44,
        min_cluster_cells=1,
    )
    panel_by_pos = {(cell.row, cell.col): "urban-panel-001" for cluster in plan.clusters[:1] for cell in cluster.cells}
    bvh_cells = _build_bvh_cells(plan, panel_by_pos, base_size_m=25, scene_size_m=800)

    assert bvh_cells
    assert _aligned_grid_origin((0, 0, 1, 1), 25) == (WEB_MERCATOR_WORLD_MIN, WEB_MERCATOR_WORLD_MIN)
    assert all(abs((cell["bboxMercator"][2] - cell["bboxMercator"][0]) - (cell["bboxMercator"][3] - cell["bboxMercator"][1])) < 1e-6 for cell in bvh_cells)
    assert all(cell["sizeM"] == 25 * 2 ** cell["layerAboveBase"] for cell in bvh_cells)

    panel = {
        "id": "urban-panel-001",
        "name": "Urban panel 001",
        "urbanScore": 0.8,
        "bboxMercator": next(cell["bboxMercator"] for cell in bvh_cells if cell["panelId"] == "urban-panel-001"),
    }
    scene_cells = _ranked_panel_cells(panel, 8, 800, base_size_m=25, grid=plan, panel_by_pos=panel_by_pos)

    assert scene_cells
    assert all(cell["sizeM"] == 800 for cell in scene_cells)
    assert all(abs((cell["bboxMercator"][2] - cell["bboxMercator"][0]) - (cell["bboxMercator"][3] - cell["bboxMercator"][1])) < 1e-6 for cell in scene_cells)

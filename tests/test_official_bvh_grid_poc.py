from __future__ import annotations

import numpy as np

from scripts.official_bvh_grid_poc import _build_bvh_grid, _max_depth_for_base


def test_bvh_parent_is_orange_until_every_leaf_is_green() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[:4, :4] = True

    _visible, all_nodes = _build_bvh_grid(mask, coarse_cells=1, max_depth=1, leaf_min_surface_ratio=0)
    root = next(cell for cell in all_nodes if cell.cell_id == "r00c00")

    assert root.status == "orange"
    assert root.leaf_count == 4
    assert root.green_leaf_count == 1


def test_bvh_parent_turns_green_only_when_all_descendant_leaves_are_green() -> None:
    mask = np.ones((8, 8), dtype=bool)

    visible, all_nodes = _build_bvh_grid(mask, coarse_cells=1, max_depth=2, leaf_min_surface_ratio=0)
    root = next(cell for cell in all_nodes if cell.cell_id == "r00c00")

    assert root.status == "green"
    assert root.leaf_count == 16
    assert root.green_leaf_count == 16
    assert visible == [root]


def test_dataset_scale_depth_reaches_25m_leaves() -> None:
    assert _max_depth_for_base(region_size_m=3200, coarse_cells=8, base_size_m=25) == 4

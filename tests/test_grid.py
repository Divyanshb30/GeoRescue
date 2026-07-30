"""The grid contract is the project's single load-bearing invariant.

`assert_matches` is what stands between a half-pixel misalignment and a
suitability map that looks entirely plausible while pointing at the wrong
ground. Nothing proved it actually fires until these tests existed.
"""

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from pipeline.grid import RESOLUTION, Grid, assert_matches, grid_for, stripes
from pipeline.regions import REGIONS


class FakeDataset:
    """Only the four attributes `assert_matches` reads."""

    def __init__(self, grid: Grid, crs=None, transform=None, shape=None):
        self.crs = crs if crs is not None else grid.crs
        self.transform = transform if transform is not None else grid.transform
        self.height, self.width = shape if shape is not None else grid.shape


# The numbers frozen in DECISIONS #008. If a refactor moves a grid by one
# pixel, every layer silently disagrees with every layer built before it.
EXPECTED = {
    "A": {"epsg": 32644, "width": 5953, "height": 6804, "origin": (186140.0, 3378690.0)},
    "B": {"epsg": 32646, "width": 7004, "height": 5603, "origin": (579260.0, 3003940.0)},
}


@pytest.mark.parametrize("region_id", sorted(REGIONS))
def test_grid_matches_frozen_geometry(region_id):
    grid = grid_for(REGIONS[region_id])
    want = EXPECTED[region_id]
    assert (grid.epsg, grid.width, grid.height) == (want["epsg"], want["width"], want["height"])
    assert (grid.transform.c, grid.transform.f) == want["origin"]


@pytest.mark.parametrize("region_id", sorted(REGIONS))
def test_grid_covers_the_whole_bbox(region_id):
    """The #012 lesson as a test.

    A UTM rectangle's corners sit outside the lat/lon bbox, so the grid must
    strictly contain the bbox's projected extent. A grid that merely touched
    it would starve its own edges - which is exactly how the land-cover build
    came back 5% nodata.
    """
    region = REGIONS[region_id]
    grid = grid_for(region)
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", f"EPSG:{region.epsg}", *region.bbox, densify_pts=21
    )
    g_left, g_bottom, g_right, g_top = grid.bounds
    assert g_left <= left and g_bottom <= bottom
    assert g_right >= right and g_top >= top


@pytest.mark.parametrize("region_id", sorted(REGIONS))
def test_pixels_are_square_and_north_up(region_id):
    t = grid_for(REGIONS[region_id]).transform
    assert t.a == RESOLUTION
    assert t.e == -RESOLUTION  # negative: rows run south, northing runs north
    assert t.b == 0 and t.d == 0  # no rotation or shear


def test_assert_matches_accepts_the_grid_itself():
    grid = grid_for(REGIONS["A"])
    assert_matches(grid, FakeDataset(grid), "identical")  # must not raise


def test_assert_matches_rejects_wrong_crs():
    grid = grid_for(REGIONS["A"])
    wrong = FakeDataset(grid, crs=CRS.from_epsg(4326))
    with pytest.raises(ValueError, match="CRS"):
        assert_matches(grid, wrong, "wrong crs")


def test_assert_matches_rejects_half_pixel_shift():
    """The failure this whole mechanism exists for.

    Five metres on a 10 m grid. Same CRS, same shape, every layer still loads,
    every map still renders - and slope no longer lines up with land cover.
    """
    grid = grid_for(REGIONS["A"])
    shifted = grid.transform * Affine.translation(0.5, 0.0)
    with pytest.raises(ValueError, match="transform"):
        assert_matches(grid, FakeDataset(grid, transform=shifted), "shifted")


def test_assert_matches_rejects_wrong_shape():
    grid = grid_for(REGIONS["A"])
    with pytest.raises(ValueError, match="shape"):
        assert_matches(grid, FakeDataset(grid, shape=(grid.height - 1, grid.width)), "short")


def test_assert_matches_reports_every_problem_at_once():
    grid = grid_for(REGIONS["A"])
    broken = FakeDataset(grid, crs=CRS.from_epsg(4326), shape=(10, 10))
    with pytest.raises(ValueError) as excinfo:
        assert_matches(grid, broken, "broken")
    assert "CRS" in str(excinfo.value) and "shape" in str(excinfo.value)


def test_stripes_tile_the_grid_exactly():
    grid = grid_for(REGIONS["A"])
    windows = stripes(grid, 512)
    covered = np.zeros(grid.height, dtype=int)
    for w in windows:
        covered[int(w.row_off) : int(w.row_off + w.height)] += 1
        assert w.width == grid.width  # full width, so reads stay contiguous
    assert (covered == 1).all()  # no gaps, no double-processing


def test_coarsened_keeps_origin_and_scales_resolution():
    grid = grid_for(REGIONS["A"])
    coarse = grid.coarsened(3, margin=0)
    assert coarse.res == RESOLUTION * 3
    assert coarse.epsg == grid.epsg
    assert (coarse.transform.c, coarse.transform.f) == (grid.transform.c, grid.transform.f)
    assert coarse.width == -(-grid.width // 3)  # ceil, so the grid is fully covered


def test_coarsened_margin_expands_outward_on_all_sides():
    grid = grid_for(REGIONS["A"])
    margin = 4
    coarse = grid.coarsened(3, margin=margin)
    step = RESOLUTION * 3
    assert coarse.transform.c == grid.transform.c - margin * step
    assert coarse.transform.f == grid.transform.f + margin * step
    assert coarse.width == -(-grid.width // 3) + 2 * margin
    assert coarse.height == -(-grid.height // 3) + 2 * margin


@pytest.mark.parametrize(
    "dtype,expected", [("float32", 3), ("float64", 3), ("uint8", 2), ("uint16", 2), ("int16", 2)]
)
def test_profile_picks_the_right_deflate_predictor(dtype, expected):
    """Predictor 2 on float data compresses badly; 3 on integers is invalid."""
    grid = grid_for(REGIONS["A"])
    assert grid.profile(1, dtype, None)["predictor"] == expected

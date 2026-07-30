"""Pure functions from the layer builders, tested against known answers.

Everything here is offline. The builders themselves need the network; these
are the parts whose correctness can be pinned down without it - and two of
them are regression tests for bugs that actually shipped.
"""

import math

import numpy as np
import pytest

from pipeline.landcover import CLASS_NAMES, MERGE, SOURCE_NAMES, apply_merge
from pipeline.osm import check_coverage, distance_metres
from pipeline.terrain import horn_slope, tile_name, tile_urls

# --------------------------------------------------------------------------
# Horn slope
# --------------------------------------------------------------------------


def test_flat_surface_has_zero_slope():
    flat = np.full((20, 20), 412.0, dtype="float32")
    assert horn_slope(flat, 30.0) == pytest.approx(0.0)


@pytest.mark.parametrize("degrees", [5.0, 15.0, 30.0, 45.0])
def test_constant_ramp_recovers_its_own_angle(degrees):
    """Horn's kernel is exact on a plane - a ramp must read back its angle."""
    res = 30.0
    gradient = math.tan(math.radians(degrees))
    cols = np.arange(20, dtype="float32")
    ramp = np.tile(cols * res * gradient, (20, 1))
    interior = horn_slope(ramp, res)[1:-1, 1:-1]
    assert interior == pytest.approx(degrees, abs=1e-4)


def test_slope_is_direction_agnostic():
    """Only the magnitude of the gradient reaches the result, never its sign."""
    res = 30.0
    cols = np.arange(20, dtype="float32")
    east = np.tile(cols * res, (20, 1))
    north = east.T
    assert horn_slope(east, res)[1:-1, 1:-1] == pytest.approx(
        horn_slope(north, res)[1:-1, 1:-1]
    )
    assert horn_slope(east, res)[1:-1, 1:-1] == pytest.approx(
        horn_slope(east[:, ::-1], res)[1:-1, 1:-1]
    )


def test_slope_scales_with_cell_size():
    """The same array over 10 m cells is three times steeper than over 30 m."""
    cols = np.arange(20, dtype="float32")
    surface = np.tile(cols * 30.0, (20, 1))
    coarse = horn_slope(surface, 30.0)[5, 5]
    fine = horn_slope(surface, 10.0)[5, 5]
    assert math.tan(math.radians(fine)) == pytest.approx(3 * math.tan(math.radians(coarse)))


# --------------------------------------------------------------------------
# DEM tile naming
# --------------------------------------------------------------------------


def test_tile_name_zero_pads_lat_two_and_lon_three():
    assert tile_name(30, 78) == "Copernicus_DSM_COG_10_N30_00_E078_00_DEM"
    assert tile_name(9, 7) == "Copernicus_DSM_COG_10_N09_00_E007_00_DEM"


def test_tile_name_handles_southern_and_western_hemispheres():
    assert "S05_00_W072_00" in tile_name(-5, -72)


def test_tile_urls_cover_every_degree_the_bbox_touches():
    urls = tile_urls((77.75, 29.90, 78.35, 30.50))
    assert len(urls) == 4  # lon 77,78 x lat 29,30
    for corner in ("N29_00_E077", "N29_00_E078", "N30_00_E077", "N30_00_E078"):
        assert any(corner in u for u in urls)


# --------------------------------------------------------------------------
# WorldCover merge
# --------------------------------------------------------------------------


def test_every_source_class_has_a_merge_target():
    """Guards the case where WorldCover gains a class and the map forgets it."""
    assert set(SOURCE_NAMES) == set(MERGE), set(SOURCE_NAMES) ^ set(MERGE)


def test_merge_targets_are_all_declared_classes():
    assert set(MERGE.values()) <= set(CLASS_NAMES)


def test_snow_maps_to_ignore_not_to_bare():
    """DECISIONS #011. 'Bare' is *ideal* ground under section 7 - routing snow
    there would score a snowfield as prime shelter siting."""
    assert MERGE[70] == 0
    assert MERGE[60] == 5  # bare stays bare


def test_apply_merge_translates_a_known_patch():
    source = np.array([[10, 20, 30], [40, 50, 60], [80, 90, 70]], dtype="uint8")
    expected = np.array([[1, 2, 2], [3, 4, 5], [6, 6, 0]], dtype="uint8")
    np.testing.assert_array_equal(apply_merge(source), expected)


def test_apply_merge_sends_unknown_codes_to_ignore():
    """An unmapped code must not become class 1 by accident."""
    assert apply_merge(np.array([[200]], dtype="uint8"))[0, 0] == 0


# --------------------------------------------------------------------------
# Distance transform
# --------------------------------------------------------------------------


def test_distance_is_zero_on_the_feature_and_grows_outward():
    mask = np.zeros((11, 11), dtype=bool)
    mask[5, 5] = True
    dist = distance_metres(mask, 10.0)
    assert dist[5, 5] == 0.0
    assert dist[5, 0] == pytest.approx(50.0)  # five 10 m cells west
    assert dist[0, 0] == pytest.approx(math.hypot(50.0, 50.0))


def test_distance_respects_cell_size():
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True
    assert distance_metres(mask, 10.0)[0, 4] == pytest.approx(40.0)
    assert distance_metres(mask, 30.0)[0, 4] == pytest.approx(120.0)


def test_empty_mask_is_an_error_not_infinity():
    with pytest.raises(SystemExit):
        distance_metres(np.zeros((5, 5), dtype=bool), 10.0)


# --------------------------------------------------------------------------
# Coverage check - the S010 regression
# --------------------------------------------------------------------------


def test_coverage_accepts_features_spread_across_the_region():
    mask = np.zeros((100, 100), dtype=bool)
    mask[::7, ::7] = True
    check_coverage("spread", mask, cells=4, min_share=0.9, fatal=True)  # must not raise


def test_coverage_rejects_features_confined_to_one_corner():
    """Exactly the S010 failure: 638 real roads, all in the north-west corner,
    producing a valid raster with a 44 km median inside a 60 km region."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[:25, :25] = True
    with pytest.raises(SystemExit, match="COVERAGE FAILED"):
        check_coverage("corner", mask, cells=4, min_share=0.9, fatal=True)


def test_coverage_only_warns_when_not_fatal(capsys):
    """Rivers legitimately miss whole quadrants, so water warns rather than fails."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[:25, :25] = True
    check_coverage("corner", mask, cells=4, min_share=0.9, fatal=False)
    assert "WARNING" in capsys.readouterr().out


def test_coverage_counts_a_cell_from_a_single_pixel():
    """One feature pixel is enough to call a cell covered - the check is about
    distribution, not density."""
    mask = np.zeros((100, 100), dtype=bool)
    for r in (10, 35, 60, 85):
        for c in (10, 35, 60, 85):
            mask[r, c] = True
    check_coverage("sparse", mask, cells=4, min_share=1.0, fatal=True)  # must not raise

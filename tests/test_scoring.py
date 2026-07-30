"""Site extraction and rendering, tested on synthetic surfaces.

The scorer itself is owner-authored and not tested here. Everything that
consumes its output is, so that when `score()` is written the only thing that
can be wrong is the scoring.
"""

import numpy as np
import pytest
from rasterio.windows import Window

from pipeline.grid import grid_for
from pipeline.regions import REGIONS
from scoring.analyze import colormap
from scoring.layers import LayerStack
from scoring.site_extraction import HECTARE_M2, clean, edge_fraction, extract, patches, summarise

SHAPE = (100, 100)


def make_stack(
    shape=SHAPE, slope=2.0, landcover=5, elevation=500.0, dist_road=200.0, dist_water=800.0
) -> LayerStack:
    """A uniform, benign stack. Tests vary only what they are testing."""
    region = REGIONS["A"]
    grid = grid_for(region)

    def full(value, dtype):
        return np.full(shape, value, dtype=dtype)

    return LayerStack(
        region=region,
        grid=grid,
        window=Window(0, 0, shape[1], shape[0]),
        slope=full(slope, "float32"),
        landcover=full(landcover, "uint8"),
        elevation=full(elevation, "float32"),
        dist_road=full(dist_road, "float32"),
        dist_water=full(dist_water, "float32"),
        valid=np.ones(shape, dtype=bool),
    )


def scores_with_block(value=80, top=10, left=10, size=20) -> np.ndarray:
    scores = np.zeros(SHAPE, dtype="uint8")
    scores[top : top + size, left : left + size] = value
    return scores


# --------------------------------------------------------------------------
# cleaning and labelling
# --------------------------------------------------------------------------


def test_clean_removes_isolated_speckle():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5, 5] = True  # lone pixel
    mask[10:15, 10:15] = True  # real patch
    cleaned = clean(mask)
    assert not cleaned[5, 5]
    assert cleaned[12, 12]


def test_patches_drop_components_below_the_size_floor():
    mask = np.zeros((40, 40), dtype=bool)
    mask[2:5, 2:5] = True  # 9 px
    mask[20:30, 20:30] = True  # 100 px
    _, keep = patches(mask, min_pixels=50)
    assert len(keep) == 1


def test_patches_join_diagonal_neighbours():
    """8-connectivity: a camp does not stop at a 45-degree step."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 2:4] = True
    mask[4:6, 4:6] = True  # touches the first block only at a corner
    _, keep = patches(mask, min_pixels=1)
    assert len(keep) == 1


def test_edge_fraction_is_higher_for_thin_shapes():
    blob = np.zeros((30, 30), dtype=bool)
    blob[5:25, 5:25] = True
    sliver = np.zeros((30, 30), dtype=bool)
    sliver[5:25, 14:16] = True
    assert edge_fraction(sliver) > edge_fraction(blob)


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def test_extract_finds_a_single_block_and_measures_its_area():
    sites = extract(scores_with_block(size=20), make_stack())
    assert len(sites) == 1
    # 20x20 px at 10 m = 40,000 m2 = 4 ha. Opening erodes nothing on a square
    # this size, so the area must come back exact.
    assert sites[0].area_ha == pytest.approx(400 * 100 / HECTARE_M2)
    assert sites[0].rank == 1


def test_extract_ranks_by_mean_score_not_by_size():
    scores = np.zeros(SHAPE, dtype="uint8")
    scores[10:40, 10:40] = 40  # big, mediocre
    scores[60:75, 60:75] = 90  # small, excellent
    sites = extract(scores, make_stack())
    assert [s.score for s in sites] == [90.0, 40.0]
    assert sites[0].area_ha < sites[1].area_ha


def test_extract_rejects_patches_under_the_minimum_area():
    # 5x5 px = 2,500 m2 = 0.25 ha, under the 0.5 ha floor
    sites = extract(scores_with_block(size=5), make_stack())
    assert sites == []


def test_extract_returns_nothing_when_everything_is_excluded():
    assert extract(np.zeros(SHAPE, dtype="uint8"), make_stack()) == []


def test_extract_ignores_pixels_marked_invalid():
    stack = make_stack()
    stack.valid[:, :] = False
    assert extract(scores_with_block(), stack) == []


def test_extract_honours_min_score():
    scores = scores_with_block(value=30)
    assert extract(scores, make_stack(), min_score=50) == []
    assert len(extract(scores, make_stack(), min_score=20)) == 1


def test_extract_caps_at_top_n():
    scores = np.zeros(SHAPE, dtype="uint8")
    for n, start in enumerate((0, 20, 40, 60, 80)):
        scores[start : start + 15, start : start + 15] = 50 + n
    sites = extract(scores, make_stack(), top_n=3)
    assert len(sites) == 3
    assert [s.rank for s in sites] == [1, 2, 3]
    assert sites[0].score > sites[-1].score


# --------------------------------------------------------------------------
# per-site statistics and flags
# --------------------------------------------------------------------------


def test_site_statistics_come_from_the_layers():
    stack = make_stack(slope=7.5, dist_road=1500.0, dist_water=250.0, elevation=640.0)
    site = extract(scores_with_block(), stack)[0]
    assert site.mean_slope_deg == pytest.approx(7.5)
    assert site.dist_road_m == 1500
    assert site.dist_water_m == 250
    assert site.mean_elevation_m == pytest.approx(640.0)
    assert site.dominant_landcover == "bare"


def test_centroid_falls_inside_the_region_grid():
    """The grid, not the bbox, is the thing a pixel centroid must lie inside.

    The synthetic block sits in the grid's north-west corner, and the grid
    deliberately overhangs the lat/lon bbox - a UTM rectangle's corners fall
    outside it (#008, #012). Asserting against the bbox would fail for a
    correct answer, so assert against the grid and check the overhang is the
    small amount that geometry predicts, not a projection blunder.
    """
    from rasterio.warp import transform as warp_transform

    site = extract(scores_with_block(), make_stack())[0]
    grid = grid_for(REGIONS["A"])

    xs, ys = warp_transform("EPSG:4326", grid.crs, [site.lon], [site.lat])
    left, bottom, right, top = grid.bounds
    assert left <= xs[0] <= right
    assert bottom <= ys[0] <= top

    lon_min, lat_min, lon_max, lat_max = REGIONS["A"].bbox
    overhang = 0.05  # degrees; the same order as the mosaic pad in #012
    assert lon_min - overhang <= site.lon <= lon_max + overhang
    assert lat_min - overhang <= site.lat <= lat_max + overhang


def test_remote_site_is_flagged():
    stack = make_stack(dist_road=7000.0, dist_water=6000.0)
    flags = " ".join(extract(scores_with_block(), stack)[0].flags)
    assert "5 km" in flags and "road" in flags and "water" in flags


def test_mostly_edge_site_is_flagged_for_the_dsm_artefact():
    """A sliver is made almost entirely of the pixels DECISIONS #010 says we
    trust least - slope and land cover both fail at patch boundaries."""
    scores = np.zeros(SHAPE, dtype="uint8")
    scores[10:70, 30:33] = 70  # long and thin: 180 px, over the area floor
    site = extract(scores, make_stack())[0]
    assert any("edge" in f for f in site.flags)


def test_compact_site_is_not_flagged_as_edge():
    site = extract(scores_with_block(size=30), make_stack())[0]
    assert not any("edge" in f for f in site.flags)


def test_site_serialises_to_plain_json_types():
    import json

    site = extract(scores_with_block(), make_stack())[0]
    json.dumps(site.as_dict())  # must not raise on numpy scalars


def test_summarise_handles_the_empty_case():
    assert "no candidate sites" in summarise([])


# --------------------------------------------------------------------------
# heatmap colours
# --------------------------------------------------------------------------


def test_excluded_is_grey_not_dark_red():
    """'Ruled out' and 'scored badly' are different statements; the map must
    not blur them."""
    lut = colormap()
    r, g, b = lut[0]
    assert abs(int(r) - int(g)) < 10 and abs(int(g) - int(b)) < 10


def test_ramp_runs_red_to_green():
    lut = colormap()
    assert lut[1][0] > lut[1][1]  # low score: red dominates
    assert lut[100][1] > lut[100][0]  # high score: green dominates


def test_green_channel_increases_with_suitability():
    lut = colormap()
    greens = [int(lut[v][1]) for v in range(1, 101)]
    assert greens[-1] > greens[0]

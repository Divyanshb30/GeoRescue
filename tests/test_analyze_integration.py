"""End-to-end run of the Gate 1 CLI against a stand-in scorer.

Everything downstream of `suitability.score()` is agent-written: layer
loading, the GeoTIFF write path, the heatmap, the sites JSON. If any of it is
broken, the owner's first run of their own scoring function would surface
*this* bug rather than theirs. So a trivial scorer is substituted here and the
whole chain is exercised on a small window of real Region A data.

Skipped automatically when the layers have not been built.
"""

import json
import runpy
import sys

import numpy as np
import pytest
import rasterio

from pipeline.grid import grid_for
from pipeline.regions import REGIONS
from scoring import layers

pytestmark = pytest.mark.skipif(
    bool(layers.missing_layers(REGIONS["A"])),
    reason=f"Region A layers not built: {layers.missing_layers(REGIONS['A'])}",
)

# A 3 x 3 km box in the Doon valley - small enough to be quick, real enough to
# contain a mix of classes.
BBOX = ["78.00", "30.25", "78.03", "30.28"]


def fake_score(stack):
    """Stand-in for the owner's scorer: obeys the contract, means nothing.

    Excludes steep ground and built/water so the exclusion path is exercised,
    then scores everything else on slope alone.
    """
    scores = np.clip(100 - stack.slope * 4, 1, 100).astype("uint8")
    scores[stack.slope > 15] = 0
    scores[np.isin(stack.landcover, (4, 6))] = 0
    scores[~stack.valid] = 0
    return scores


@pytest.fixture
def run_cli(monkeypatch, tmp_path):
    """Run analyze.main() with the fake scorer, writing into a temp dir."""

    def _run(extra_args=()):
        from scoring import analyze, suitability

        monkeypatch.setattr(suitability, "score", fake_score)
        monkeypatch.setattr(analyze.suitability, "score", fake_score)
        monkeypatch.setattr(
            sys, "argv",
            ["analyze", "--region", "A", "--bbox", *BBOX,
             "--out-dir", str(tmp_path), *extra_args],
        )
        analyze.main()
        return tmp_path

    return _run


def test_cli_writes_all_three_outputs(run_cli):
    out = run_cli()
    for name in ("suitability.tif", "suitability_heatmap.png", "sites.json"):
        assert (out / name).exists(), f"{name} was not written"
        assert (out / name).stat().st_size > 0


def test_suitability_raster_is_georeferenced_and_on_the_region_crs(run_cli):
    out = run_cli()
    grid = grid_for(REGIONS["A"])
    with rasterio.open(out / "suitability.tif") as src:
        assert src.crs == grid.crs
        assert src.dtypes[0] == "uint8"
        assert src.nodata == 0
        assert src.res == (grid.res, grid.res)
        data = src.read(1)
    assert data.max() <= 100


def test_windowed_output_stays_aligned_to_the_parent_grid(run_cli):
    """A clipped raster must still sit on the region grid - its origin has to
    be a whole number of pixels from the grid origin, or the window silently
    shifts everything it contains."""
    out = run_cli()
    grid = grid_for(REGIONS["A"])
    with rasterio.open(out / "suitability.tif") as src:
        col = (src.transform.c - grid.transform.c) / grid.res
        row = (grid.transform.f - src.transform.f) / grid.res
    assert col == pytest.approx(round(col))
    assert row == pytest.approx(round(row))


def test_sites_json_carries_the_honest_labelling_caveats(run_cli):
    """Spec §3: every report states what its numbers do and do not mean."""
    payload = json.loads((run_cli() / "sites.json").read_text())
    assert payload["region"] == "A"
    assert payload["imagery_window"] == "2026-01-01/2026-04-30"
    blob = " ".join(payload["caveats"]).lower()
    for required in ("composite", "proxy", "straight-line", "not certified"):
        assert required in blob, f"caveat missing: {required}"


def test_every_reported_site_number_traces_to_a_layer(run_cli):
    """The zero-hallucination rule at the data level: a site's statistics must
    be consistent with the rasters, not decorative."""
    payload = json.loads((run_cli() / "sites.json").read_text())
    stack = layers.load("A", layers.window_for_bounds(
        grid_for(REGIONS["A"]),
        rasterio.warp.transform_bounds(
            "EPSG:4326", grid_for(REGIONS["A"]).crs, *[float(v) for v in BBOX], densify_pts=21
        ),
    ))
    for site in payload["sites"]:
        assert 0 < site["score"] <= 100
        assert site["area_ha"] >= 0.5
        assert site["mean_slope_deg"] <= 15.0  # the fake scorer excluded steeper ground
        assert stack.slope.min() <= site["mean_slope_deg"] <= stack.slope.max()
        assert site["dominant_landcover"] not in ("built", "water+wetland")


def test_heatmap_is_a_readable_png(run_cli):
    out = run_cli()
    with rasterio.open(out / "suitability_heatmap.png") as src:
        assert src.count == 3
        assert src.width > 0 and src.height > 0


def test_module_entrypoint_is_wired(monkeypatch, tmp_path):
    """`python -m scoring.analyze` must actually reach main()."""
    from scoring import analyze

    monkeypatch.setattr(sys, "argv", ["analyze", "--region", "A", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("scoring.analyze", run_name="__main__")
    assert excinfo.value.code == 0


def test_analyze_output_feeds_the_report_generator_cleanly(run_cli):
    """The whole product path: layers -> score -> sites -> prose, with every
    numeral in the prose traced back to a computed statistic."""
    from reports.template import render, verify

    payload = json.loads((run_cli() / "sites.json").read_text())
    text = render(payload)

    assert verify(text, payload) == [], "report invented a number"
    assert "Region A" in text
    assert "not certified site planning" in text.lower()

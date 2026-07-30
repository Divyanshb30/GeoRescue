"""Polygon in, suitability heatmap and ranked sites out. The Gate 1 CLI.

This is the walking skeleton the product ships on: it runs the whole chain -
load aligned layers, score them, extract candidate sites, render a heatmap -
against the WorldCover baseline, before any model exists. When the U-Net
lands it swaps in behind a toggle and nothing else here changes.

Run:  python -m scoring.analyze --region A
      python -m scoring.analyze --region A --bbox 78.00 30.25 78.10 30.35
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import rasterio.shutil
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds

from pipeline.grid import grid_for
from pipeline.regions import REGIONS
from scoring import layers, site_extraction, suitability

# Low suitability reads red, high reads green - the one colour convention a
# responder does not have to be taught. Excluded pixels are deliberately a
# flat dark grey, not dark red: "ruled out" is a different statement from
# "scored badly", and the map must not blur the two.
EXCLUDED_COLOUR = (38, 38, 42)
RAMP = [
    (0, (158, 44, 44)),
    (35, (196, 118, 44)),
    (60, (206, 186, 62)),
    (80, (118, 168, 68)),
    (100, (38, 116, 62)),
]


def colormap() -> np.ndarray:
    """256-entry RGB lookup: index 0 excluded, 1-100 the red-to-green ramp."""
    lut = np.zeros((256, 3), dtype="uint8")
    lut[0] = EXCLUDED_COLOUR
    stops = np.array([s for s, _ in RAMP], dtype="float32")
    colours = np.array([c for _, c in RAMP], dtype="float32")
    for value in range(1, 101):
        for channel in range(3):
            lut[value, channel] = np.interp(value, stops, colours[:, channel])
    lut[101:] = lut[100]
    return lut


def render_heatmap(scores: np.ndarray, out_path: Path, width: int = 1400) -> None:
    lut = colormap()
    height, source_width = scores.shape
    step = max(source_width // width, 1)
    small = scores[::step, ::step]  # nearest: these are scores, not radiance
    rgb = lut[small].transpose(2, 0, 1)
    profile = {
        "driver": "GTiff", "width": small.shape[1], "height": small.shape[0],
        "count": 3, "dtype": "uint8",
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as tmp:
            tmp.write(rgb)
            rasterio.shutil.copy(tmp, out_path, driver="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=sorted(REGIONS))
    parser.add_argument(
        "--bbox", nargs=4, type=float, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="analyse a sub-area; default is the whole region",
    )
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--min-score", type=int, default=1)
    parser.add_argument(
        "--out-dir", type=Path,
        help="where to write results; defaults to the region's layer directory",
    )
    args = parser.parse_args()

    region = REGIONS[args.region]
    grid = grid_for(region)
    out_dir = args.out_dir or layers.region_dir(region)
    out_dir.mkdir(parents=True, exist_ok=True)

    window = None
    if args.bbox:
        bounds = transform_bounds("EPSG:4326", grid.crs, *args.bbox, densify_pts=21)
        window = layers.window_for_bounds(grid, bounds)
        area_km2 = (window.width * window.height) * grid.res**2 / 1e6
        print(f"window {int(window.width)} x {int(window.height)} px = {area_km2:.1f} km2")

    print(f"Region {region.id}: loading layers")
    stack = layers.load(region.id, window)
    print(f"  {stack.shape[1]} x {stack.shape[0]} px, {stack.valid.mean() * 100:.2f}% valid")

    try:
        scores = suitability.score(stack)
    except NotImplementedError as exc:
        raise SystemExit(
            f"\nsuitability.score() is not written yet.\n  {exc}\n\n"
            "Everything else in this chain is ready: layers load and validate, "
            "site extraction and rendering are tested.\nWrite score() and this "
            "command produces a heatmap and a ranked shortlist."
        )

    scores = np.asarray(scores)
    if scores.shape != stack.shape:
        raise SystemExit(f"score() returned {scores.shape}, expected {stack.shape}")
    if scores.dtype != np.uint8:
        print(f"  note: score() returned {scores.dtype}, casting to uint8")
        scores = scores.astype("uint8")

    excluded = (scores == 0) & stack.valid
    print(f"  excluded {excluded.mean() * 100:.1f}% of valid pixels")
    scored = scores[scores > 0]
    if scored.size:
        print(
            f"  suitability p50={np.percentile(scored, 50):.0f} "
            f"p90={np.percentile(scored, 90):.0f} max={scored.max()}"
        )

    profile = grid.profile(1, "uint8", 0)
    if window is not None:
        profile.update(
            width=int(window.width), height=int(window.height), transform=stack.transform()
        )
    raster_path = out_dir / "suitability.tif"
    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(scores, 1)
        dst.set_band_description(1, "suitability_0_100")

    heatmap_path = out_dir / "suitability_heatmap.png"
    render_heatmap(scores, heatmap_path)

    sites = site_extraction.extract(
        scores, stack, top_n=args.top, min_score=args.min_score
    )
    print("\n" + site_extraction.summarise(sites))

    sites_path = out_dir / "sites.json"
    sites_path.write_text(
        json.dumps(
            {
                "region": region.id,
                "imagery_window": "2026-01-01/2026-04-30",
                "landcover_source": "ESA WorldCover 2021 v200",
                "dem_source": "Copernicus GLO-30",
                "caveats": [
                    "Composite imagery, Jan-Apr 2026 - not a single date.",
                    "Flood exposure is a proxy from elevation and water distance, "
                    "not a hydrological model.",
                    "Distances are straight-line, not along-road.",
                    "Thresholds are v1 defaults, not certified site-planning standards.",
                ],
                "sites": [s.as_dict() for s in sites],
            },
            indent=2,
        )
    )

    print(f"\n{raster_path}\n{heatmap_path}\n{sites_path}")


if __name__ == "__main__":
    main()

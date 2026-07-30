"""ESA WorldCover 2021 merged to the project's six classes, on the region grid.

Two jobs this layer does. It is the **baseline** the product ships on before
the trained model exists (spec section 10 — the walking-skeleton order), and it
is the **weak supervision** the U-Net trains against later. Both uses want the
same six classes, so the merge happens once, here.

WorldCover ships eleven classes at 10 m in geographic coordinates. Merging to
six is a deliberate cut (spec section 8): scoring only ever distinguishes
these six, and eleven classes would slow convergence for legend detail nobody
reads. Resampling is nearest-neighbour throughout — averaging class codes 40
and 60 would invent class 50.

Run:  python -m pipeline.landcover --region A
"""

import argparse
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.merge import merge

from pipeline.grid import assert_matches, grid_for, to_grid
from pipeline.regions import REGIONS, Region

WORLDCOVER_BUCKET = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
TILE_DEG = 3  # WorldCover tiles are 3x3 degrees, named by south-west corner

# The region grid is a rectangle in UTM, which is a curved quadrilateral in
# lat/lon - its corners sit outside the lat/lon bbox. Mosaicking to the bbox
# exactly therefore starves the grid edges; measured at 5.0% nodata before
# this pad existed.
BBOX_PAD_DEG = 0.05

SOURCE_NAMES = {
    10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland",
    50: "built-up", 60: "bare / sparse", 70: "snow & ice",
    80: "permanent water", 90: "herbaceous wetland", 95: "mangroves",
    100: "moss & lichen",
}

# The project's six, frozen here (spec section 8). 0 is reserved for
# nodata/ignore - it is the label a segmentation loss skips, so anything that
# cannot be honestly assigned goes there rather than into a wrong bucket.
CLASS_NAMES = {
    0: "nodata/ignore",
    1: "trees",
    2: "shrub + grassland",
    3: "cropland",
    4: "built-up",
    5: "bare / sparse",
    6: "water + wetland",
}

MERGE = {
    10: 1,
    20: 2, 30: 2,
    40: 3,
    50: 4,
    60: 5, 100: 5,   # moss & lichen behaves like sparse ground for siting
    80: 6, 90: 6, 95: 6,
    # 70 snow & ice is deliberately NOT merged. "Bare" is *ideal* terrain under
    # the section 7 rules, so folding snow into it would score a glacier as
    # prime shelter ground; folding it into water would exclude it for a reason
    # that is not true. It maps to 0 and is excluded and ignored, on purpose.
    70: 0,
}

NODATA = 0
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
}


def tile_urls(bbox: tuple[float, float, float, float]) -> list[str]:
    lon_min, lat_min, lon_max, lat_max = bbox

    def floor3(v: float) -> int:
        return int(math.floor(v / TILE_DEG) * TILE_DEG)

    urls = []
    for lat in range(floor3(lat_min), floor3(lat_max) + 1, TILE_DEG):
        for lon in range(floor3(lon_min), floor3(lon_max) + 1, TILE_DEG):
            ns = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            ew = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            urls.append(f"{WORLDCOVER_BUCKET}/ESA_WorldCover_10m_2021_v200_{ns}{ew}_Map.tif")
    return urls


def load_mosaic(region: Region):
    lon_min, lat_min, lon_max, lat_max = region.bbox
    bounds = (
        lon_min - BBOX_PAD_DEG, lat_min - BBOX_PAD_DEG,
        lon_max + BBOX_PAD_DEG, lat_max + BBOX_PAD_DEG,
    )
    opened = []
    for url in tile_urls(bounds):  # padded, so a pad crossing a tile edge still resolves
        try:
            opened.append(rasterio.open(url))
        except RasterioIOError:
            print(f"  no tile (all ocean?), skipping: {url.rsplit('/', 1)[-1]}")
    if not opened:
        raise SystemExit("no WorldCover tiles available for this bbox")
    try:
        mosaic, transform = merge(opened, bounds=bounds)
        crs, nodata = opened[0].crs, opened[0].nodata
    finally:
        for src in opened:
            src.close()
    print(f"  mosaicked {len(opened)} tile(s) -> {mosaic.shape[1]} x {mosaic.shape[2]} px in {crs}")
    return mosaic[0], transform, crs, nodata


def apply_merge(source: np.ndarray) -> np.ndarray:
    """Map the eleven source codes to the six project classes via a lookup."""
    lut = np.zeros(256, dtype="uint8")
    for src_code, dst_code in MERGE.items():
        lut[src_code] = dst_code
    return lut[source]


def histogram(array: np.ndarray, names: dict[int, str], title: str) -> dict[int, float]:
    print(f"  {title}")
    total = array.size
    shares = {}
    for code, count in sorted(zip(*np.unique(array, return_counts=True))):
        pct = count / total * 100
        shares[int(code)] = pct
        print(f"    {int(code):>3} {names.get(int(code), '?'):<18} {pct:6.2f}%  {'#' * int(pct / 2)}")
    return shares


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=sorted(REGIONS))
    args = parser.parse_args()

    region = REGIONS[args.region]
    grid = grid_for(region)
    out_dir = DATA_DIR / f"region{region.id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Region {region.id}: WorldCover -> {grid.width} x {grid.height} @ {grid.res:.0f} m")
    with rasterio.Env(**GDAL_ENV):
        mosaic, src_transform, src_crs, src_nodata = load_mosaic(region)

    source = to_grid(
        mosaic, src_transform, src_crs, src_nodata, grid, Resampling.nearest, "uint8", 0
    )
    histogram(source, SOURCE_NAMES, "source classes (WorldCover 11):")

    merged = apply_merge(source)
    shares = histogram(merged, CLASS_NAMES, "merged classes (project 6):")

    out_path = out_dir / "landcover.tif"
    with rasterio.open(out_path, "w", **grid.profile(1, "uint8", NODATA)) as dst:
        dst.write(merged, 1)
        dst.set_band_description(1, "landcover_6class")
        dst.update_tags(**{f"class_{k}": v for k, v in CLASS_NAMES.items()})

    with rasterio.open(out_path) as src:
        assert_matches(grid, src, out_path.name)

    missing = [CLASS_NAMES[c] for c in range(1, 7) if shares.get(c, 0) == 0]
    if missing:
        print(f"  WARNING: classes absent from this region: {missing}")
    else:
        print("  all six classes present")
    print(f"  grid check: PASS\n{out_path}")


if __name__ == "__main__":
    main()

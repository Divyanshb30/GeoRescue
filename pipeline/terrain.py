"""Elevation and slope for a region, on the region grid.

Copernicus GLO-30 ships as 1x1 degree COG tiles in geographic coordinates.
Two things have to happen before that is usable here: the tiles covering the
bbox get mosaicked, and the result gets projected into the region's UTM CRS -
because slope is a ratio of rise to run, and a "run" measured in degrees is
not a distance.

Slope is computed with Horn's method at the DEM's own ~30 m posting, then
resampled to the 10 m grid. Not the other way round: upsampling first and
differencing after would measure the gradient of the interpolator between real
postings, and bilinear upsampling in particular produces flat facets with
discontinuous edges - visible in the output as blocky slope.

Run:  python -m pipeline.terrain --region A
"""

import argparse
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.merge import merge
from rasterio.warp import reproject

from pipeline.grid import Grid, assert_matches, grid_for
from pipeline.regions import REGIONS, Region

DEM_BUCKET = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
DEM_FACTOR = 3  # 10 m grid -> ~30 m, GLO-30's real posting (1 arcsec)
DEM_MARGIN = 4  # coarse cells of context so Horn's 3x3 never runs off real data
BBOX_PAD_DEG = 0.05  # ~5 km, so the margin has source data to reproject from

NODATA = -9999.0

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
}


def tile_name(lat: int, lon: int) -> str:
    """GLO-30 tiles are named by their south-west corner, zero-padded."""
    ns = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
    ew = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def tile_urls(bbox: tuple[float, float, float, float]) -> list[str]:
    lon_min, lat_min, lon_max, lat_max = bbox
    urls = []
    for lat in range(math.floor(lat_min), math.floor(lat_max) + 1):
        for lon in range(math.floor(lon_min), math.floor(lon_max) + 1):
            name = tile_name(lat, lon)
            urls.append(f"{DEM_BUCKET}/{name}/{name}.tif")
    return urls


def load_mosaic(region: Region) -> tuple[np.ndarray, object, object]:
    """Merge the GLO-30 tiles covering the padded bbox, still in EPSG:4326."""
    lon_min, lat_min, lon_max, lat_max = region.bbox
    bounds = (
        lon_min - BBOX_PAD_DEG, lat_min - BBOX_PAD_DEG,
        lon_max + BBOX_PAD_DEG, lat_max + BBOX_PAD_DEG,
    )
    opened = []
    for url in tile_urls(region.bbox):
        try:
            opened.append(rasterio.open(url))
        except RasterioIOError:
            # Tiles over open ocean are simply absent from the bucket. Land
            # regions should never hit this; say so rather than fail silently.
            print(f"  missing DEM tile (ocean?), skipping: {url.rsplit('/', 1)[-1]}")
    if not opened:
        raise SystemExit("no DEM tiles available for this bbox")
    try:
        mosaic, transform = merge(opened, bounds=bounds)
        crs, nodata = opened[0].crs, opened[0].nodata
    finally:
        for src in opened:
            src.close()
    print(f"  mosaicked {len(opened)} tile(s) -> {mosaic.shape[1]} x {mosaic.shape[2]} px in {crs}")
    return mosaic[0], transform, (crs, nodata)


def to_grid(
    source: np.ndarray, src_transform, src_crs, src_nodata, grid: Grid, resampling: Resampling
) -> np.ndarray:
    out = np.full(grid.shape, NODATA, dtype="float32")
    reproject(
        source=source,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=NODATA,
        resampling=resampling,
    )
    return out


def horn_slope(elevation: np.ndarray, res: float) -> np.ndarray:
    """Slope in degrees, Horn (1981) - the 3x3 kernel ArcGIS and GDAL both use.

    Each partial derivative is a weighted difference across the whole 3x3
    neighbourhood, with the middle row/column counted twice. Fitting the plane
    to eight neighbours rather than two makes it far less twitchy on noisy
    DEMs than a simple central difference.
    """
    z = np.pad(elevation, 1, mode="edge")
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]

    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * res)
    # Rows increase southward, so this is the north-south gradient with the
    # sign flipped. Irrelevant here: only its magnitude reaches the result.
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * res)

    return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype("float32")


def write(path: Path, grid: Grid, array: np.ndarray, description: str) -> None:
    with rasterio.open(path, "w", **grid.profile(1, "float32", NODATA)) as dst:
        dst.write(array, 1)
        dst.set_band_description(1, description)


def summarise(name: str, array: np.ndarray, unit: str, cuts: tuple[float, ...] = ()) -> None:
    valid = array[array != NODATA]
    pcts = " ".join(f"p{q}={np.percentile(valid, q):.1f}" for q in (1, 25, 50, 75, 99))
    print(f"  {name}: {pcts} {unit}  (min {valid.min():.1f}, max {valid.max():.1f})")
    for cut in cuts:
        print(f"      {(valid > cut).mean() * 100:5.1f}% above {cut:g}{unit}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=sorted(REGIONS))
    args = parser.parse_args()

    region = REGIONS[args.region]
    grid = grid_for(region)
    coarse = grid.coarsened(DEM_FACTOR, margin=DEM_MARGIN)
    out_dir = DATA_DIR / f"region{region.id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Region {region.id}: DEM -> {grid.width} x {grid.height} @ {grid.res:.0f} m")
    with rasterio.Env(**GDAL_ENV):
        mosaic, src_transform, (src_crs, src_nodata) = load_mosaic(region)

    # Elevation goes straight to the analysis grid - one interpolation, cubic
    # for a smooth surface. The flood proxy needs it as a height, not a slope.
    elevation = to_grid(mosaic, src_transform, src_crs, src_nodata, grid, Resampling.cubic)

    # Slope takes the detour through native posting, per the module docstring.
    coarse_elev = to_grid(mosaic, src_transform, src_crs, src_nodata, coarse, Resampling.bilinear)
    coarse_slope = horn_slope(coarse_elev, coarse.res)
    coarse_slope[coarse_elev == NODATA] = NODATA
    slope = to_grid(
        coarse_slope, coarse.transform, coarse.crs, NODATA, grid, Resampling.bilinear
    )

    dem_path, slope_path = out_dir / "dem.tif", out_dir / "slope.tif"
    write(dem_path, grid, elevation, "elevation_m")
    write(slope_path, grid, slope, "slope_deg")

    summarise("elevation", elevation, " m")
    summarise("slope", slope, " deg", cuts=(5, 10, 15))

    for path in (dem_path, slope_path):
        with rasterio.open(path) as src:
            assert_matches(grid, src, path.name)
    print(f"  grid check: PASS\n{dem_path}\n{slope_path}")


if __name__ == "__main__":
    main()

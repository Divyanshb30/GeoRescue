"""Distance-to-road and distance-to-water rasters from OpenStreetMap.

Two of the five section 7 factors are proximity questions: can a truck reach
the site, and is there water near it. Both become the same shape of layer - a
raster where each pixel holds the straight-line distance in metres to the
nearest feature of interest - so the scorer just reads a number instead of
doing geometry at request time.

Source is a Geofabrik extract, not a live Overpass query, for the same reason
scenes are pinned to manifests (DECISIONS #007): a live query returns
something slightly different every run, and a result that cannot be
reproduced cannot be defended.

Distances are Euclidean, not along-network. A site 800 m from a road across a
ravine reads as 800 m. Stated as a limitation rather than fixed - routing is
a different project.

Run:  python -m pipeline.osm --region A
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
import requests
from rasterio.features import rasterize
from scipy import ndimage

from pipeline.grid import Grid, assert_matches, grid_for
from pipeline.regions import REGIONS, Region

GEOFABRIK = "https://download.geofabrik.de/asia/india"

# Distances up to ~5 km carry weight in section 7, so features just outside the
# bbox still matter to pixels near the edge. Pad well past that: without it,
# every border pixel would report the distance to the nearest feature *inside*
# the box and read as more remote than it is.
BBOX_PAD_DEG = 0.1  # ~11 km

# What a relief truck can drive on. Deliberately excludes footway, path,
# cycleway, steps, bridleway and pedestrian: those are access for a person,
# not for the vehicle that delivers shelter materials.
ROAD_VALUES = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "service", "track",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}

# Perennial water only. Streams, drains and ditches are seasonal across this
# monsoon-driven landscape, and a dry ditch is not a water supply - counting
# them would make half the Doon valley look well-served in April.
WATERWAY_VALUES = {"river", "canal"}
WATER_AREA_TAGS = {"natural": {"water"}, "landuse": {"reservoir", "basin"}}

NODATA = -1.0  # distances are never negative
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OSM_DIR = DATA_DIR / "osm"

# Keeping the OSM driver's node index in RAM instead of spilling to a temp file
# is the difference between a minute and many for a 200 MB extract.
GDAL_ENV = {"OSM_MAX_TMPFILE_SIZE": "1024", "OGR_INTERLEAVED_READING": "YES"}


def ensure_extract(region: Region) -> Path:
    """Download the region's Geofabrik zone extract once, then reuse it."""
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    path = OSM_DIR / f"{region.osm_zone}-latest.osm.pbf"
    if path.exists():
        print(f"  extract cached: {path.name} ({path.stat().st_size / 1e6:.0f} MB)")
        return path

    url = f"{GEOFABRIK}/{region.osm_zone}-latest.osm.pbf"
    print(f"  downloading {url}")
    t0 = time.time()
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"  downloaded {path.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f}s")
    return path


def padded_bbox(region: Region) -> tuple[float, float, float, float]:
    lon_min, lat_min, lon_max, lat_max = region.bbox
    return (
        lon_min - BBOX_PAD_DEG, lat_min - BBOX_PAD_DEG,
        lon_max + BBOX_PAD_DEG, lat_max + BBOX_PAD_DEG,
    )


def read_layer(path: Path, layer: str, bbox, columns: list[str]):
    """Read one OSM layer inside the bbox, keeping only the columns we filter on."""
    t0 = time.time()
    gdf = pyogrio.read_dataframe(
        path, layer=layer, bbox=bbox, columns=columns, use_arrow=False
    )
    print(f"    {layer}: {len(gdf):,} features in {time.time() - t0:.0f}s")
    return gdf


def read_roads(path: Path, bbox):
    gdf = read_layer(path, "lines", bbox, ["highway"])
    return gdf[gdf["highway"].isin(ROAD_VALUES)]


def read_water(path: Path, bbox):
    """Rivers and canals as lines, plus lakes and reservoirs as polygons."""
    lines = read_layer(path, "lines", bbox, ["waterway"])
    lines = lines[lines["waterway"].isin(WATERWAY_VALUES)]

    areas = read_layer(path, "multipolygons", bbox, ["natural", "landuse"])
    keep = np.zeros(len(areas), dtype=bool)
    for column, values in WATER_AREA_TAGS.items():
        keep |= areas[column].isin(values).to_numpy()
    areas = areas[keep]

    print(f"    water: {len(lines):,} linear + {len(areas):,} areal")
    return lines, areas


def burn(grid: Grid, *frames) -> np.ndarray:
    """Rasterize geometries onto the grid as a boolean presence mask.

    all_touched because a river drawn as a one-dimensional line would
    otherwise fall between pixel centres and vanish in places.
    """
    mask = np.zeros(grid.shape, dtype="uint8")
    for frame in frames:
        if frame.empty:
            continue
        geoms = frame.to_crs(grid.crs).geometry
        rasterize(
            ((geom, 1) for geom in geoms if geom is not None),
            out=mask,
            transform=grid.transform,
            all_touched=True,
            merge_alg=rasterio.enums.MergeAlg.replace,
        )
    return mask.astype(bool)


def check_coverage(name: str, mask: np.ndarray, cells: int, min_share: float, fatal: bool) -> None:
    """Fail if features only occupy a corner of the region.

    Exists because of a real failure: the first build used Geofabrik's northern
    zone for Region A, which stops at the Himachal border. It returned 638 real
    roads - all of them in the bbox's north-west corner - and produced a
    perfectly well-formed distance raster claiming a 44 km median distance to
    road inside a 60 km-wide region containing three cities. Nothing crashed.
    A distribution check is the only thing that catches a wrong-but-valid input.
    """
    rows = np.array_split(mask, cells, axis=0)
    occupied = sum(1 for row in rows for cell in np.array_split(row, cells, axis=1) if cell.any())
    share = occupied / (cells * cells)
    verdict = f"{name}: features present in {occupied}/{cells * cells} cells ({share:.0%})"
    if share >= min_share:
        print(f"  coverage ok - {verdict}")
        return
    message = f"{verdict}, below the {min_share:.0%} floor - wrong extract, or a bad tag filter?"
    if fatal:
        raise SystemExit(f"  COVERAGE FAILED - {message}")
    print(f"  WARNING - {message}")


def distance_metres(mask: np.ndarray, res: float) -> np.ndarray:
    """Euclidean distance from every pixel to the nearest True pixel, in metres."""
    if not mask.any():
        raise SystemExit("nothing rasterized - every distance would be infinite")
    # EDT measures distance to the nearest *zero*, so the mask is inverted:
    # feature pixels become 0 and get distance 0, as intended.
    return ndimage.distance_transform_edt(~mask, sampling=res).astype("float32")


def write(path: Path, grid: Grid, array: np.ndarray, description: str) -> None:
    with rasterio.open(path, "w", **grid.profile(1, "float32", NODATA)) as dst:
        dst.write(array, 1)
        dst.set_band_description(1, description)


def summarise(name: str, dist: np.ndarray, cuts: tuple[int, ...]) -> None:
    pcts = " ".join(f"p{q}={np.percentile(dist, q) / 1000:.2f}" for q in (50, 90, 99))
    print(f"  {name}: {pcts} km (max {dist.max() / 1000:.1f} km)")
    for cut in cuts:
        print(f"      {(dist <= cut).mean() * 100:5.1f}% within {cut / 1000:g} km")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=sorted(REGIONS))
    args = parser.parse_args()

    region = REGIONS[args.region]
    grid = grid_for(region)
    out_dir = DATA_DIR / f"region{region.id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Region {region.id}: OSM -> {grid.width} x {grid.height} @ {grid.res:.0f} m")
    path = ensure_extract(region)
    bbox = padded_bbox(region)

    with rasterio.Env(**GDAL_ENV):
        roads = read_roads(path, bbox)
        water_lines, water_areas = read_water(path, bbox)

    print(f"    roads: {len(roads):,} drivable ways")

    # Roads must blanket a populated region, so a gap is an error. Rivers
    # legitimately miss whole quadrants, so that check only warns.
    for name, mask, cuts, min_share, fatal in (
        ("dist_road", burn(grid, roads), (500, 1000, 5000), 0.90, True),
        ("dist_water", burn(grid, water_lines, water_areas), (500, 1000, 5000), 0.50, False),
    ):
        covered = mask.mean() * 100
        check_coverage(name, mask, cells=4, min_share=min_share, fatal=fatal)
        dist = distance_metres(mask, grid.res)
        out_path = out_dir / f"{name}.tif"
        write(out_path, grid, dist, name)
        print(f"  {name}: {covered:.2f}% of pixels are feature pixels")
        summarise(name, dist, cuts)
        with rasterio.open(out_path) as src:
            assert_matches(grid, src, out_path.name)

    print(f"  grid check: PASS\n{out_dir / 'dist_road.tif'}\n{out_dir / 'dist_water.tif'}")


if __name__ == "__main__":
    main()

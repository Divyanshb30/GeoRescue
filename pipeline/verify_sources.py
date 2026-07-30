"""Day-1 data-source verification for GeoRescue.

Confirms every external data path works from this machine before any other
pipeline code gets written. Read-only and tiny: each probe reads a small
window straight over HTTP; nothing is written to disk.

Run:  python -m pipeline.verify_sources
"""

import sys

import requests
import rasterio
from pystac_client import Client
from rasterio.windows import Window

from pipeline.regions import REGIONS

# Imported, never retyped: DECISIONS #004 makes `pipeline/regions.py` the only
# place a bbox may live. A verifier probing a stale copy of the bbox would
# report all-clear for ground the pipeline no longer looks at.
BBOX_A = REGIONS["A"].bbox

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
S2_COLLECTIONS = ("sentinel-2-c1-l2a", "sentinel-2-l2a")
IMAGERY_WINDOW = "2026-01-01/2026-04-30"  # pre-monsoon: monsoon scenes are cloud-wrecked
MAX_CLOUD_PCT = 30

# Static tiles covering the bbox centre: DEM tiles are 1x1 deg named by SW
# corner; WorldCover tiles are 3x3 deg on a 3-deg grid.
DEM_TILE_URL = (
    "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/"
    "Copernicus_DSM_COG_10_N30_00_E078_00_DEM/"
    "Copernicus_DSM_COG_10_N30_00_E078_00_DEM.tif"
)
WORLDCOVER_TILE_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_N30E078_Map.tif"
)
GEOFABRIK_URL = "https://download.geofabrik.de/asia/india-latest.osm.pbf"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

WORLDCOVER_CLASSES = {
    10: "trees", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
    60: "bare", 70: "snow", 80: "water", 90: "wetland", 95: "mangrove",
    100: "moss",
}

# Stops GDAL listing the whole S3 "directory" on open — one HEAD plus range
# reads instead, which is the entire point of COGs.
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}


def read_probe_window(url, size=256):
    """Read a size x size window from the centre of a remote COG."""
    with rasterio.Env(**GDAL_ENV), rasterio.open(url) as src:
        row = max((src.height - size) // 2, 0)
        col = max((src.width - size) // 2, 0)
        data = src.read(1, window=Window(col, row, size, size))
        return data, str(src.crs), (src.height, src.width)


def check_sentinel2_stac():
    client = Client.open(EARTH_SEARCH_URL)
    lines, best = [], None
    for coll in S2_COLLECTIONS:
        search = client.search(
            collections=[coll],
            bbox=BBOX_A,
            datetime=IMAGERY_WINDOW,
            query={"eo:cloud_cover": {"lt": MAX_CLOUD_PCT}},
        )
        n = search.matched()
        lines.append(f"{coll}: {n} scenes <{MAX_CLOUD_PCT}% cloud")
        if best is None and n:
            best = (coll, next(search.items()))
    if best is None:
        raise RuntimeError("; ".join(lines) + " — nothing found in window")
    coll, item = best
    asset = item.assets.get("red") or item.assets.get("B04")
    data, crs, shape = read_probe_window(asset.href)
    lines.append(
        f"probe read ok: scene {item.id} band red, {data.shape[0]}x{data.shape[1]} px, "
        f"DN {int(data.min())}-{int(data.max())}, full grid {shape[0]}x{shape[1]}, {crs}"
    )
    return "\n      ".join(lines)


def check_copernicus_dem():
    data, crs, shape = read_probe_window(DEM_TILE_URL, size=128)
    return (
        f"tile N30E078 read ok: elevation {data.min():.0f}-{data.max():.0f} m, "
        f"grid {shape[0]}x{shape[1]}, {crs}"
    )


def check_worldcover():
    data, crs, shape = read_probe_window(WORLDCOVER_TILE_URL, size=256)
    seen = sorted(set(data.ravel().tolist()) - {0})
    names = ", ".join(WORLDCOVER_CLASSES.get(v, str(v)) for v in seen)
    return f"tile N30E078 read ok: classes in probe window [{names}], {crs}"


def check_osm_geofabrik():
    r = requests.head(GEOFABRIK_URL, allow_redirects=True, timeout=30)
    r.raise_for_status()
    size_gb = int(r.headers.get("Content-Length", 0)) / 1e9
    return f"india-latest.osm.pbf reachable, {size_gb:.2f} GB"


def check_osm_overpass():
    # Overpass usage policy: anonymous/default user-agents get 406'd.
    headers = {"User-Agent": "georescue/0.1 (student project)"}
    query = '[out:json][timeout:25];way["highway"](30.30,78.00,30.33,78.03);out count;'
    r = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=40)
    r.raise_for_status()
    n_ways = r.json()["elements"][0]["tags"]["ways"]
    return f"query ok: {n_ways} highway ways in a 3x3 km probe box near Dehradun"


CHECKS = [
    ("Sentinel-2 L2A via Earth Search STAC", check_sentinel2_stac),
    ("Copernicus GLO-30 DEM (AWS)", check_copernicus_dem),
    ("ESA WorldCover 2021 v200 (AWS)", check_worldcover),
    ("OSM extract (Geofabrik)", check_osm_geofabrik),
    ("OSM live query (Overpass)", check_osm_overpass),
]


def main():
    failures = 0
    for name, fn in CHECKS:
        try:
            detail = fn()
            print(f"[ OK ] {name}\n      {detail}")
        except Exception as exc:  # noqa: BLE001 — report every failure, keep going
            failures += 1
            print(f"[FAIL] {name}\n      {type(exc).__name__}: {exc}")
    print("-" * 60)
    print("all sources verified" if failures == 0 else f"{failures} source(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

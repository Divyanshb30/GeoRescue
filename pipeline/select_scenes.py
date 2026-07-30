"""Select the Sentinel-2 scenes that feed a region's composite.

Queries Earth Search for scenes in the imagery window (DECISIONS.md #001,
#002), shortlists on scene-level cloud cover, keeps the clearest scenes per
MGRS tile, and pins the result to a JSON manifest so the composite build is
reproducible even if the catalog changes underneath us.

Run:  python -m pipeline.select_scenes --region A
"""

import argparse
import calendar
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from pystac import Item
from pystac_client import Client

from pipeline.regions import REGIONS, Region

# pystac tries to migrate the deprecated storage extension on every c1 item
# and warns that https hrefs carry no s3 bucket to parse. Benign: we use the
# hrefs exactly as served. Silence only that message.
warnings.filterwarnings("ignore", message="Could not parse bucket/account")

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-c1-l2a"  # DECISIONS.md #002
DATE_RANGE = "2026-01-01/2026-04-30"  # DECISIONS.md #001
MAX_SCENE_CLOUD_PCT = 30.0
MAX_SCENES_PER_TILE = 12

# swir16/swir22 recorded for the optional B11/B12 experiment (spec section 8);
# scl is the per-pixel quality layer the composite step masks with.
BANDS = ("blue", "green", "red", "nir", "swir16", "swir22", "scl")

MANIFEST_DIR = Path(__file__).parent / "manifests"


def tile_of(item: Item) -> str:
    code = item.properties.get("grid:code")  # c1 style: "MGRS-43RGP"
    if code:
        return code.removeprefix("MGRS-")
    tile = item.properties.get("s2:mgrs_tile")  # legacy-collection style
    if tile:
        return tile
    raise KeyError(f"{item.id}: no MGRS tile property")


def epsg_of(item: Item) -> int:
    if "proj:epsg" in item.properties:
        return int(item.properties["proj:epsg"])
    code = item.properties.get("proj:code", "")  # newer projection-ext style
    if code.startswith("EPSG:"):
        return int(code.removeprefix("EPSG:"))
    raise KeyError(f"{item.id}: no EPSG property")


def search_scenes(client: Client, region: Region) -> list[Item]:
    search = client.search(
        collections=[COLLECTION],
        bbox=region.bbox,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": MAX_SCENE_CLOUD_PCT}},
    )
    return list(search.items())


def check_tile_coverage(client: Client, region: Region, kept_tiles: set[str]) -> None:
    """Warn if the cloud filter starved an entire tile.

    The unfiltered catalog's tiles over the bbox are gapless ground coverage;
    a tile present there but absent after filtering is a hole in the
    composite, not just fewer looks.
    """
    search = client.search(collections=[COLLECTION], bbox=region.bbox, datetime=DATE_RANGE)
    all_tiles = {tile_of(item) for item in search.items()}
    starved = sorted(all_tiles - kept_tiles)
    if starved:
        print(
            f"WARNING: cloud filter left no scenes for tile(s) {starved} - "
            f"composite would have a hole there; raise MAX_SCENE_CLOUD_PCT"
        )
    else:
        print(f"tile coverage ok: all {len(all_tiles)} catalog tiles over this bbox survived the filter")


def select_per_tile(items: list[Item]) -> dict[str, list[Item]]:
    """Clearest MAX_SCENES_PER_TILE scenes per tile, returned in date order.

    Grouped per tile rather than one global top-N so a persistently cloudy
    tile still gets enough looks instead of being starved by clearer ones.
    """
    by_tile: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        by_tile[tile_of(item)].append(item)

    selection = {}
    for tile, tile_items in sorted(by_tile.items()):
        tile_items.sort(key=lambda i: i.properties["eo:cloud_cover"])
        keep = tile_items[:MAX_SCENES_PER_TILE]
        selection[tile] = sorted(keep, key=lambda i: i.datetime)
    return selection


def scene_record(item: Item) -> dict:
    return {
        "id": item.id,
        "datetime": item.properties["datetime"],
        "tile": tile_of(item),
        "epsg": epsg_of(item),
        "cloud_pct": item.properties["eo:cloud_cover"],
        "nodata_pct": item.properties.get("s2:nodata_pixel_percentage"),
        "assets": {band: item.assets[band].href for band in BANDS},
    }


def build_manifest(region: Region, selection: dict[str, list[Item]]) -> dict:
    scenes = [scene_record(i) for items in selection.values() for i in items]
    return {
        "region": region.id,
        "collection": COLLECTION,
        "date_range": DATE_RANGE,
        "max_scene_cloud_pct": MAX_SCENE_CLOUD_PCT,
        "max_scenes_per_tile": MAX_SCENES_PER_TILE,
        "scene_count": len(scenes),
        "scenes": scenes,
    }


def print_report(region: Region, selection: dict[str, list[Item]]) -> None:
    print(
        f"Region {region.id} ({region.name}) - {COLLECTION}, {DATE_RANGE}, "
        f"scene cloud < {MAX_SCENE_CLOUD_PCT:.0f}%"
    )
    for tile, items in selection.items():
        clouds = [i.properties["eo:cloud_cover"] for i in items]
        months = Counter(i.datetime.month for i in items)
        spread = " ".join(f"{calendar.month_abbr[m]}x{n}" for m, n in sorted(months.items()))
        print(
            f"  {tile} (EPSG:{epsg_of(items[0])}): {len(items):2d} scenes, "
            f"cloud {min(clouds):4.1f}-{max(clouds):4.1f}%, {spread}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=sorted(REGIONS))
    args = parser.parse_args()
    region = REGIONS[args.region]

    client = Client.open(STAC_URL)
    items = search_scenes(client, region)
    if not items:
        raise SystemExit("search returned no scenes - check window and cloud threshold")
    selection = select_per_tile(items)
    print_report(region, selection)
    check_tile_coverage(client, region, set(selection))

    manifest = build_manifest(region, selection)
    MANIFEST_DIR.mkdir(exist_ok=True)
    out_path = MANIFEST_DIR / f"region{region.id}_scenes.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n{manifest['scene_count']} scenes pinned -> {out_path}")


if __name__ == "__main__":
    main()

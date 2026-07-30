"""Build a region's cloud-free Sentinel-2 composite from its pinned manifest.

One clear scene does not exist: every Sentinel-2 pass over the Himalayan
foothills carries cloud, cloud shadow, or snow somewhere. So instead of
picking a best scene we take many, throw away the bad pixels in each using the
scene's own quality layer (SCL), and reduce what survives to a per-pixel
median. A pixel obscured on 2026-02-14 is clear on 2026-03-05, and the median
of its clear looks is a better estimate of the ground than any single date.

Median, not mean: a cloud edge the SCL missed is a bright outlier, and the
median ignores outliers where the mean drags toward them.

Nothing is downloaded whole. Each scene is presented through a WarpedVRT - a
virtual view that reprojects and resamples it onto the region grid on the fly
- and only the stripe being worked on is fetched, as HTTP range reads.

Run:  python -m pipeline.composite --region A
      python -m pipeline.composite --region A --stripes 1   (quick smoke test)
"""

import argparse
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from pipeline.grid import Grid, grid_for, stripe_bounds, stripes
from pipeline.regions import REGIONS, Region

REFLECTANCE_BANDS = ("blue", "green", "red", "nir")  # B2, B3, B4, B8 - all 10 m native
QA_BAND = "scl"

# Sentinel-2 Scene Classification Layer, the per-pixel verdict L2A processing
# ships with each scene. Keep only pixels it calls actual ground:
#   4 vegetation   5 not-vegetated   6 water   7 unclassified
# Dropped: 0 nodata, 1 defective, 2 dark/cast-shadow, 3 cloud shadow,
#   8 cloud medium-prob, 9 cloud high-prob, 10 thin cirrus, 11 snow/ice.
# 7 "unclassified" is kept deliberately - it is SCL shrugging, not SCL flagging
# cloud, and dropping it punches holes in otherwise fine terrain.
SCL_KEEP = frozenset({4, 5, 6, 7})

# 0 is the fill value in L2A reflectance COGs; true surface reflectance is
# never exactly 0, so it doubles as the nodata test.
FILL = 0

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MANIFEST_DIR = Path(__file__).parent / "manifests"

STRIPE_HEIGHT = 512  # px; ~370 MB peak for 24 scenes at Region A's width

# Each read is mostly waiting on S3, not computing, so reads run concurrently.
# Safe because every (scene, band) is its own dataset handle touched by one
# thread at a time - GDAL forbids sharing one dataset across threads, not
# using many datasets at once.
READ_WORKERS = 8

# /vsicurl tuning. Without DISABLE_READDIR_ON_OPEN, GDAL lists the whole S3
# prefix on every open; the retry settings absorb the transient 5xx that a
# multi-gigabyte run over a home connection will otherwise die on.
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "CPL_VSIL_CURL_CACHE_SIZE": "268435456",  # 256 MB of fetched blocks kept warm
    "VSI_CACHE": "TRUE",
}


def load_manifest(region: Region) -> dict:
    path = MANIFEST_DIR / f"region{region.id}_scenes.json"
    if not path.exists():
        raise SystemExit(f"no manifest at {path} - run: python -m pipeline.select_scenes --region {region.id}")
    return json.loads(path.read_text())


def open_on_grid(href: str, grid: Grid, categorical: bool) -> tuple[rasterio.DatasetReader, WarpedVRT]:
    """Open a remote band and present it already reprojected onto the grid.

    Bilinear for reflectance (continuous, and tile 43RGP genuinely changes UTM
    zone here); nearest for SCL, because averaging class codes 8 and 4 would
    invent a class 6.

    Returns both handles so the caller can close them; opening happens off the
    main thread, and ExitStack is not thread-safe.
    """
    with rasterio.Env(**GDAL_ENV):
        src = rasterio.open(href)
        vrt = WarpedVRT(
            src,
            crs=grid.crs,
            transform=grid.transform,
            width=grid.width,
            height=grid.height,
            resampling=Resampling.nearest if categorical else Resampling.bilinear,
            src_nodata=FILL,
            nodata=FILL,
        )
    return src, vrt


def source_bounds_on_grid(vrt: WarpedVRT, grid: Grid) -> tuple[float, float, float, float]:
    """The underlying scene's real footprint, in grid coordinates.

    The VRT spans the whole region by construction, so its own bounds say
    nothing about where the scene actually has pixels. This does, and it lets
    us skip stripes a scene does not touch instead of paying for an HTTP read
    that returns fill.
    """
    src = vrt.src_dataset
    return transform_bounds(src.crs, grid.crs, *src.bounds, densify_pts=21)


def read_retry(vrt: WarpedVRT, window: Window, attempts: int = 3) -> np.ndarray:
    # rasterio.Env is thread-local, so worker threads need the /vsicurl
    # settings re-entered or they fall back to GDAL's chatty defaults.
    with rasterio.Env(**GDAL_ENV):
        for attempt in range(attempts):
            try:
                return vrt.read(1, window=window)
            except RasterioIOError:
                if attempt == attempts - 1:
                    raise
                time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def read_band(pool: ThreadPoolExecutor, vrts: list[WarpedVRT], window: Window) -> list[np.ndarray]:
    """Same window from many scenes at once."""
    return list(pool.map(lambda vrt: read_retry(vrt, window), vrts))


def intersects(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def composite_stripe(
    pool: ThreadPoolExecutor,
    scenes: list[dict],
    window: Window,
    stripe: tuple[float, float, float, float],
    keep_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Median of the clear looks in one stripe.

    Returns (4 x h x w reflectance uint16, h x w clear-look count uint8).
    """
    height, width = int(window.height), int(window.width)

    live = [s for s in scenes if intersects(s["bounds"], stripe)]
    masks, keep_scenes = [], []
    for scene, scl in zip(live, read_band(pool, [s["vrts"][QA_BAND] for s in live], window)):
        mask = keep_codes[scl]  # lookup table indexed by class code -> bool
        if mask.any():
            masks.append(mask)
            keep_scenes.append(scene)

    clear_count = (
        np.count_nonzero(masks, axis=0).astype("uint8")
        if masks
        else np.zeros((height, width), dtype="uint8")
    )

    out = np.zeros((len(REFLECTANCE_BANDS), height, width), dtype="uint16")
    if not keep_scenes:
        return out, clear_count

    values = np.empty((len(keep_scenes), height, width), dtype="float32")
    for b, band in enumerate(REFLECTANCE_BANDS):
        reads = read_band(pool, [s["vrts"][band] for s in keep_scenes], window)
        for k, (arr, mask) in enumerate(zip(reads, masks)):
            # NaN marks "no usable look here"; nanmedian then ignores it per
            # pixel, so a pixel clear in 3 of 12 scenes still gets a value.
            values[k] = np.where(mask & (arr != FILL), arr, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels
            median = np.nanmedian(values, axis=0)
        out[b] = np.where(np.isnan(median), FILL, np.rint(median)).astype("uint16")

    return out, clear_count


def progress_path(out_path: Path) -> Path:
    return out_path.with_suffix(".progress.json")


def load_progress(out_path: Path, signature: dict) -> set[int]:
    """Stripe indices already written, but only if the run matches.

    A resumed run that used a different scene set or stripe height would stitch
    two incompatible halves together, so any signature change starts over.
    """
    path = progress_path(out_path)
    if not (path.exists() and out_path.exists()):
        return set()
    saved = json.loads(path.read_text())
    if saved.get("signature") != signature:
        print("previous run used different settings - starting over")
        return set()
    return set(saved.get("done", []))


def save_progress(out_path: Path, signature: dict, done: set[int]) -> None:
    progress_path(out_path).write_text(
        json.dumps({"signature": signature, "done": sorted(done)}, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=sorted(REGIONS))
    parser.add_argument("--stripe-height", type=int, default=STRIPE_HEIGHT)
    parser.add_argument("--stripes", type=int, help="process only the first N stripes (smoke test)")
    parser.add_argument("--max-scenes", type=int, help="use only the first N scenes (smoke test)")
    args = parser.parse_args()

    region = REGIONS[args.region]
    grid = grid_for(region)
    manifest = load_manifest(region)
    scenes = manifest["scenes"][: args.max_scenes]

    out_dir = DATA_DIR / f"region{region.id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    comp_path = out_dir / "s2_composite.tif"
    count_path = out_dir / "s2_clear_count.tif"

    windows = stripes(grid, args.stripe_height)
    bounds_by_index = [stripe_bounds(grid, w) for w in windows]
    todo_indices = range(len(windows)) if args.stripes is None else range(min(args.stripes, len(windows)))

    signature = {
        "scene_ids": [s["id"] for s in scenes],
        "stripe_height": args.stripe_height,
        "shape": list(grid.shape),
        "scl_keep": sorted(SCL_KEEP),
    }
    done = load_progress(comp_path, signature)

    keep_codes = np.zeros(256, dtype=bool)
    keep_codes[list(SCL_KEEP)] = True

    print(
        f"Region {region.id}: {len(scenes)} scenes -> {grid.width} x {grid.height} px, "
        f"{len(windows)} stripes of {args.stripe_height} rows"
    )

    with rasterio.Env(**GDAL_ENV), ExitStack() as stack:
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=READ_WORKERS))
        t0 = time.time()
        for scene in scenes:
            scene["vrts"] = {}
        jobs = [
            (scene, band)
            for scene in scenes
            for band in (*REFLECTANCE_BANDS, QA_BAND)
        ]
        opened = pool.map(
            lambda job: open_on_grid(job[0]["assets"][job[1]], grid, categorical=(job[1] == QA_BAND)),
            jobs,
        )
        for (scene, band), (src, vrt) in zip(jobs, opened):
            stack.callback(src.close)
            stack.callback(vrt.close)
            scene["vrts"][band] = vrt
        for scene in scenes:
            scene["bounds"] = source_bounds_on_grid(scene["vrts"][QA_BAND], grid)
        print(f"  opened {len(jobs)} remote bands in {time.time() - t0:.0f}s")

        fresh = not done
        comp_profile = grid.profile(len(REFLECTANCE_BANDS), "uint16", FILL)
        count_profile = grid.profile(1, "uint8", 0)
        comp = stack.enter_context(rasterio.open(comp_path, "w" if fresh else "r+", **(comp_profile if fresh else {})))
        counts = stack.enter_context(rasterio.open(count_path, "w" if fresh else "r+", **(count_profile if fresh else {})))
        if fresh:
            for b, band in enumerate(REFLECTANCE_BANDS, 1):
                comp.set_band_description(b, band)
            counts.set_band_description(1, "clear_looks")

        for i in todo_indices:
            if i in done:
                continue
            window = windows[i]
            t0 = time.time()
            block, clear = composite_stripe(pool, scenes, window, bounds_by_index[i], keep_codes)
            comp.write(block, window=window)
            counts.write(clear, 1, window=window)
            done.add(i)
            save_progress(comp_path, signature, done)
            holes = float((clear == 0).mean() * 100)
            print(
                f"  stripe {i + 1}/{len(windows)} rows {int(window.row_off)}-"
                f"{int(window.row_off + window.height)}: median {int(np.median(clear))} clear looks, "
                f"{holes:.2f}% empty, {time.time() - t0:.0f}s"
            )

    print(f"\ncomposite -> {comp_path}")
    print(f"clear-look count -> {count_path}")
    if len(done) < len(windows):
        print(f"{len(windows) - len(done)} stripes still to do - rerun the same command to resume")


if __name__ == "__main__":
    main()

"""Load a region's factor layers as one aligned, validated stack.

The grid contract (DECISIONS #008) means these five rasters are already
pixel-for-pixel comparable. This module is where that promise is *checked*
rather than assumed - every layer is asserted against `grid_for(region)` on
the way in, so the scorer downstream can index them together without a single
defensive line.

Loading is windowed: an API request covering 25 km2 reads 25 km2, not the
whole 40-megapixel region.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

from pipeline.grid import Grid, assert_matches, grid_for
from pipeline.regions import REGIONS, Region

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# name -> (filename, band). Order is the order the scorer sees them.
LAYER_FILES = {
    "slope": ("slope.tif", 1),          # degrees
    "landcover": ("landcover.tif", 1),  # 6-class codes, 0 = ignore
    "elevation": ("dem.tif", 1),        # metres
    "dist_road": ("dist_road.tif", 1),  # metres
    "dist_water": ("dist_water.tif", 1),  # metres
}


@dataclass(frozen=True)
class LayerStack:
    """Five factor layers over the same window, plus what they mean.

    Every array has identical shape. `valid` is False wherever any layer has
    no data, so the scorer never has to ask whether a pixel is real.
    """

    region: Region
    grid: Grid
    window: Window
    slope: np.ndarray
    landcover: np.ndarray
    elevation: np.ndarray
    dist_road: np.ndarray
    dist_water: np.ndarray
    valid: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.slope.shape

    def transform(self):
        """Affine transform for this window, for writing results back out."""
        return rasterio.windows.transform(self.window, self.grid.transform)


def region_dir(region: Region) -> Path:
    return DATA_DIR / f"region{region.id}"


def missing_layers(region: Region) -> list[str]:
    return [
        name
        for name, (filename, _) in LAYER_FILES.items()
        if not (region_dir(region) / filename).exists()
    ]


def window_for_bounds(grid: Grid, bounds: tuple[float, float, float, float]) -> Window:
    """Grid window covering bounds (in the region's CRS), clipped to the grid."""
    window = from_bounds(*bounds, transform=grid.transform).round_offsets().round_lengths()
    full = Window(0, 0, grid.width, grid.height)
    return window.intersection(full)


def load(region_id: str, window: Window | None = None) -> LayerStack:
    """Read every factor layer over `window`, asserting each sits on the grid."""
    region = REGIONS[region_id]
    grid = grid_for(region)

    absent = missing_layers(region)
    if absent:
        raise SystemExit(
            f"region {region.id} is missing {absent} - "
            f"build them first (pipeline.terrain / pipeline.landcover / pipeline.osm)"
        )

    if window is None:
        window = Window(0, 0, grid.width, grid.height)

    arrays = {}
    for name, (filename, band) in LAYER_FILES.items():
        path = region_dir(region) / filename
        with rasterio.open(path) as src:
            # The whole point of the contract: check here, trust everywhere else.
            assert_matches(grid, src, path.name)
            data = src.read(band, window=window)
            nodata = src.nodata
        arrays[name] = data
        arrays.setdefault("_nodata", {})[name] = nodata

    nodata = arrays.pop("_nodata")
    valid = np.ones(arrays["slope"].shape, dtype=bool)
    for name, data in arrays.items():
        if nodata[name] is not None:
            valid &= data != nodata[name]

    return LayerStack(region=region, grid=grid, window=window, valid=valid, **arrays)

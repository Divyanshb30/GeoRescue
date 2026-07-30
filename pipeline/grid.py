"""The one grid every raster in a region must land on.

Sentinel-2, the DEM, WorldCover and the OSM distance layers all arrive in
different CRSs, resolutions and extents. Rather than align them pairwise
whenever two layers meet, each region gets a single canonical grid here, and
every stage in the pipeline resamples onto it once, on write. Downstream code
may then treat any two layers of a region as pixel-for-pixel comparable
without checking - which is exactly the assumption the scorer makes when it
multiplies a slope raster by a land-cover raster.

`assert_matches` is the enforcement: any layer that does not sit exactly on
the grid fails loudly at load rather than silently producing a shifted
suitability map (spec section 13, the CRS/alignment risk).

Run:  python -m pipeline.grid
"""

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds

from pipeline.regions import REGIONS, Region

# Sentinel-2's native resolution for B2/B3/B4/B8. Every layer is resampled to
# it: upsampling the 30 m DEM is honest (it adds no detail, it just shares the
# grid), downsampling S2 to 30 m would throw away the resolution the whole
# project is built on.
RESOLUTION = 10.0


@dataclass(frozen=True)
class Grid:
    """A fixed raster geometry: where the pixels are and how many there are."""

    epsg: int
    transform: Affine  # maps (col, row) -> (x, y) in the region's UTM CRS
    width: int
    height: int

    @property
    def crs(self) -> CRS:
        return CRS.from_epsg(self.epsg)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def res(self) -> float:
        """Pixel size in metres, read off the transform rather than assumed."""
        return self.transform.a

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left, top = self.transform.c, self.transform.f
        return left, top - self.height * self.res, left + self.width * self.res, top

    def coarsened(self, factor: int, margin: int = 0) -> "Grid":
        """Same origin and CRS, `factor`x larger pixels, optional margin of them.

        Used where a source layer's real resolution is coarser than 10 m: a
        gradient is honest at the posting spacing it was measured at, and
        upsampling first would compute it across interpolated values. The
        margin gives neighbourhood operators real data to read at the edges
        instead of clamping.
        """
        res = self.res * factor
        left, _, _, top = self.bounds
        return Grid(
            epsg=self.epsg,
            transform=Affine(res, 0.0, left - margin * res, 0.0, -res, top + margin * res),
            width=math.ceil(self.width / factor) + 2 * margin,
            height=math.ceil(self.height / factor) + 2 * margin,
        )

    def profile(self, count: int, dtype: str, nodata: float | None) -> dict:
        """Creation options for a layer written on this grid.

        Tiled + deflate is the COG-friendly layout: readers fetch one 512 px
        tile instead of a whole row, which is what makes windowed reads over
        HTTP cheap later when these live on GCS.
        """
        return {
            "driver": "GTiff",
            "crs": self.crs,
            "transform": self.transform,
            "width": self.width,
            "height": self.height,
            "count": count,
            "dtype": dtype,
            "nodata": nodata,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "deflate",
            # 2 = horizontal differencing (integers), 3 = float predictor.
            # Using 2 on float data silently compresses worse, not wrongly.
            "predictor": 3 if np.dtype(dtype).kind == "f" else 2,
            "BIGTIFF": "IF_SAFER",
        }


def grid_for(region: Region) -> Grid:
    """Build a region's canonical grid from its frozen bbox (DECISIONS #004)."""
    # densify_pts samples along the bbox edges instead of only its corners: a
    # lat/lon rectangle reprojects to a slightly curved quadrilateral, and the
    # corner-only bounds would clip the bulge in the middle of an edge.
    left, bottom, right, top = transform_bounds(
        "EPSG:4326", f"EPSG:{region.epsg}", *region.bbox, densify_pts=21
    )

    # Snap outward to whole multiples of the resolution, so pixel edges land on
    # round metre coordinates and two regions' grids stay commensurate.
    left = math.floor(left / RESOLUTION) * RESOLUTION
    bottom = math.floor(bottom / RESOLUTION) * RESOLUTION
    right = math.ceil(right / RESOLUTION) * RESOLUTION
    top = math.ceil(top / RESOLUTION) * RESOLUTION

    return Grid(
        epsg=region.epsg,
        transform=Affine(RESOLUTION, 0.0, left, 0.0, -RESOLUTION, top),
        width=round((right - left) / RESOLUTION),
        height=round((top - bottom) / RESOLUTION),
    )


def to_grid(
    source: np.ndarray,
    src_transform: Affine,
    src_crs,
    src_nodata,
    grid: Grid,
    resampling: Resampling,
    dtype: str,
    dst_nodata,
) -> np.ndarray:
    """Reproject an array onto the grid. The only way layers get here."""
    out = np.full(grid.shape, dst_nodata, dtype=dtype)
    reproject(
        source=source,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=dst_nodata,
        resampling=resampling,
    )
    return out


def assert_matches(grid: Grid, src, name: str) -> None:
    """Raise unless an opened dataset sits exactly on the grid."""
    problems = []
    if src.crs != grid.crs:
        problems.append(f"CRS {src.crs} != {grid.crs}")
    if not src.transform.almost_equals(grid.transform, precision=1e-6):
        problems.append(f"transform {src.transform!r} != {grid.transform!r}")
    if (src.height, src.width) != grid.shape:
        problems.append(f"shape {(src.height, src.width)} != {grid.shape}")
    if problems:
        raise ValueError(f"{name} is off-grid: " + "; ".join(problems))


def stripes(grid: Grid, height: int) -> list[Window]:
    """Full-width horizontal slabs covering the grid, top to bottom.

    Stripes rather than square blocks: a source COG stores its data in rows of
    tiles, so a full-width read pulls contiguous ranges and costs far fewer
    HTTP requests than the same area taken as a column of squares.
    """
    return [
        Window(0, row, grid.width, min(height, grid.height - row))
        for row in range(0, grid.height, height)
    ]


def stripe_bounds(grid: Grid, window: Window) -> tuple[float, float, float, float]:
    """A window's extent in the region's CRS (left, bottom, right, top)."""
    return tuple(window_bounds(window, grid.transform))


def _describe(region: Region) -> Iterator[str]:
    grid = grid_for(region)
    left, bottom, right, top = grid.bounds
    yield f"Region {region.id} ({region.name}) - EPSG:{grid.epsg} @ {RESOLUTION:.0f} m"
    yield f"  size    {grid.width} x {grid.height} px = {grid.width * grid.height / 1e6:.1f} Mpx"
    yield f"  extent  x {left:.0f} -> {right:.0f} m, y {bottom:.0f} -> {top:.0f} m"
    yield f"  ground  {(right - left) / 1000:.1f} x {(top - bottom) / 1000:.1f} km"
    yield f"  origin  {grid.transform.c:.0f}, {grid.transform.f:.0f} (top-left pixel corner)"


def main() -> None:
    for region in REGIONS.values():
        for line in _describe(region):
            print(line)


if __name__ == "__main__":
    main()

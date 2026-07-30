"""Turn a suitability surface into a ranked shortlist of candidate sites.

A per-pixel score is not an answer. "Pitch the camp here" needs a *place*: a
contiguous patch big enough to hold something, reachable, and describable in
a sentence. This module does that reduction - threshold, clean, label, filter
by area, rank, and report per-site statistics that trace back to the layers.

Deliberately independent of how the score was computed, so it can rank the
WorldCover-direct baseline and the model-driven score with identical logic.
"""

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage

from pipeline.grid import Grid
from scoring.layers import LayerStack

MIN_AREA_HA = 0.5  # spec section 7; 0.5 ha is a small camp, not a tent
HECTARE_M2 = 10_000.0

# A patch one pixel wide is a rasterisation artifact, not a site. Opening with
# a 3x3 structuring element removes those without eroding real patches much.
OPENING_STRUCTURE = np.ones((3, 3), dtype=bool)

CLASS_NAMES = {
    0: "nodata", 1: "trees", 2: "shrub+grass", 3: "cropland",
    4: "built", 5: "bare", 6: "water+wetland",
}


@dataclass(frozen=True)
class Site:
    """One candidate site. Every field traces to a computed raster statistic."""

    rank: int
    lat: float
    lon: float
    score: float          # mean suitability over the patch
    score_max: float
    area_ha: float
    mean_slope_deg: float
    max_slope_deg: float
    dist_road_m: float    # nearest point of the patch to a road
    dist_water_m: float
    mean_elevation_m: float
    dominant_landcover: str
    flags: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def clean(mask: np.ndarray) -> np.ndarray:
    """Drop speckle so connected components are places, not noise."""
    return ndimage.binary_opening(mask, structure=OPENING_STRUCTURE)


def patches(mask: np.ndarray, min_pixels: int) -> tuple[np.ndarray, list[int]]:
    """Label connected components and keep those at or above the size floor.

    8-connectivity: a diagonal step still makes one patch. A camp does not
    stop at a 45-degree boundary.
    """
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        return labels, []
    sizes = ndimage.sum_labels(mask, labels, index=range(1, count + 1))
    keep = [i + 1 for i, size in enumerate(sizes) if size >= min_pixels]
    return labels, keep


def edge_fraction(patch: np.ndarray) -> float:
    """Share of a patch's pixels that touch its boundary.

    Matters because of the DSM problem (DECISIONS #010): slope is inflated by
    canopy and roof edges, and land cover is least reliable at those same
    boundaries. A patch that is nearly all edge is mostly made of the pixels
    we trust least.
    """
    interior = ndimage.binary_erosion(patch, structure=OPENING_STRUCTURE)
    total = patch.sum()
    return float((total - interior.sum()) / total) if total else 1.0


def describe(
    label_id: int,
    labels: np.ndarray,
    scores: np.ndarray,
    stack: LayerStack,
    grid: Grid,
    transform,
) -> dict:
    """Per-site statistics, all computed - none assumed."""
    patch = labels == label_id
    rows, cols = np.nonzero(patch)

    # Centroid in pixel space -> the region's CRS -> lon/lat for the UI.
    row_c, col_c = rows.mean(), cols.mean()
    x, y = transform * (col_c + 0.5, row_c + 0.5)
    lon, lat = _to_lonlat(x, y, grid)

    area_ha = patch.sum() * (grid.res**2) / HECTARE_M2
    landcover = stack.landcover[patch]
    codes, counts = np.unique(landcover, return_counts=True)
    dominant = int(codes[counts.argmax()])

    flags = []
    if edge_fraction(patch) > 0.5:
        flags.append("mostly-edge: slope and land cover least reliable here (DSM artefact)")
    if stack.dist_road[patch].min() > 5000:
        flags.append("over 5 km from the nearest drivable road")
    if stack.dist_water[patch].min() > 5000:
        flags.append("over 5 km from perennial water")
    if area_ha < 1.0:
        flags.append("small site: under 1 ha")

    return {
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "score": round(float(scores[patch].mean()), 1),
        "score_max": round(float(scores[patch].max()), 1),
        "area_ha": round(float(area_ha), 2),
        "mean_slope_deg": round(float(stack.slope[patch].mean()), 2),
        "max_slope_deg": round(float(stack.slope[patch].max()), 2),
        "dist_road_m": round(float(stack.dist_road[patch].min())),
        "dist_water_m": round(float(stack.dist_water[patch].min())),
        "mean_elevation_m": round(float(stack.elevation[patch].mean()), 1),
        "dominant_landcover": CLASS_NAMES.get(dominant, str(dominant)),
        "flags": flags,
    }


def _to_lonlat(x: float, y: float, grid: Grid) -> tuple[float, float]:
    from rasterio.warp import transform as warp_transform

    lons, lats = warp_transform(grid.crs, "EPSG:4326", [x], [y])
    return lons[0], lats[0]


def extract(
    scores: np.ndarray,
    stack: LayerStack,
    top_n: int = 5,
    min_score: int = 1,
    min_area_ha: float = MIN_AREA_HA,
) -> list[Site]:
    """Top-N candidate sites, ranked by mean suitability over the patch.

    `min_score=1` means "anything not excluded", since the scorer reserves 0
    for exclusions. Raise it to shortlist only strong ground.
    """
    grid = stack.grid
    min_pixels = max(int(round(min_area_ha * HECTARE_M2 / grid.res**2)), 1)

    mask = clean((scores >= min_score) & stack.valid)
    labels, keep = patches(mask, min_pixels)
    if not keep:
        return []

    transform = stack.transform()
    described = [describe(i, labels, scores, stack, grid, transform) for i in keep]
    described.sort(key=lambda d: (d["score"], d["area_ha"]), reverse=True)

    return [Site(rank=n, **d) for n, d in enumerate(described[:top_n], start=1)]


def summarise(sites: Iterable[Site]) -> str:
    sites = list(sites)
    if not sites:
        return "no candidate sites met the minimum area and score"
    lines = [f"{len(sites)} candidate site(s):"]
    for s in sites:
        lines.append(
            f"  #{s.rank}  score {s.score:5.1f}  {s.area_ha:6.2f} ha  "
            f"slope {s.mean_slope_deg:4.1f} deg  road {s.dist_road_m / 1000:4.1f} km  "
            f"water {s.dist_water_m / 1000:4.1f} km  {s.dominant_landcover}"
        )
        for flag in s.flags:
            lines.append(f"        ! {flag}")
    return "\n".join(lines)

"""Region definitions, frozen in DECISIONS.md #004.

Every pipeline stage imports these; nothing else may hardcode a bbox or CRS.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    id: str
    name: str  # short slug, used in file and folder names
    bbox: tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max (EPSG:4326)
    epsg: int  # the one UTM CRS every raster of this region lives in


REGIONS: dict[str, Region] = {
    "A": Region(id="A", name="dehradun", bbox=(77.75, 29.90, 78.35, 30.50), epsg=32644),
    "B": Region(id="B", name="majuli", bbox=(93.80, 26.65, 94.50, 27.15), epsg=32646),
}

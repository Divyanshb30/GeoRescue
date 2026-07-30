"""Render any layer on the region grid as a small PNG you can actually look at.

Every stage of this pipeline writes a 40-megapixel raster that no viewer opens
comfortably and no test really checks. Statistics say a composite is plausible;
only an image says the Ganga is in the right place. This is the eyeball step,
kept as a script so each new layer gets one for free.

Run:  python -m pipeline.quicklook data/regionA/s2_composite.tif --bands red green blue
      python -m pipeline.quicklook data/regionA/s2_clear_count.tif
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import rasterio
import rasterio.shutil
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile

STRETCH = (2.0, 98.0)  # percentile clip; raw reflectance renders near-black otherwise

# Class codes are labels, not quantities: stretching them would imply water is
# "more" than trees. Categorical layers get a fixed palette instead, so two
# runs are also directly comparable - which matters when the model's output is
# put next to the WorldCover baseline.
CLASS_COLOURS = {
    0: (0, 0, 0),          # nodata / ignore
    1: (26, 92, 45),       # trees
    2: (163, 186, 72),     # shrub + grassland
    3: (232, 178, 92),     # cropland
    4: (198, 62, 48),      # built-up
    5: (176, 168, 148),    # bare / sparse
    6: (44, 102, 194),     # water + wetland
}

# A quicklook is a picture, not a layer: dropping the transform is the point,
# so rasterio's "this has no geotransform" warning is noise here.
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def resolve_bands(src, names: list[str] | None) -> list[int]:
    if not names:
        return [1] if src.count == 1 else [1, 2, 3]
    indices = []
    for name in names:
        if name.isdigit():
            indices.append(int(name))
            continue
        if name not in src.descriptions:
            raise SystemExit(f"no band named {name!r}; file has {src.descriptions}")
        indices.append(src.descriptions.index(name) + 1)
    return indices


def stretch_to_byte(band: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Percentile-clip one band to 0-255, computed over valid pixels only."""
    if not valid.any():
        return np.zeros(band.shape, dtype="uint8")
    lo, hi = np.percentile(band[valid], STRETCH)
    if hi <= lo:
        hi = lo + 1
    scaled = (band.astype("float32") - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype("uint8")


def render(
    path: Path, band_names: list[str] | None, width: int, out_path: Path, categorical: bool
) -> None:
    with rasterio.open(path) as src:
        indices = resolve_bands(src, band_names)
        height = max(round(src.height * width / src.width), 1)
        # Averaging on the way down, not nearest: a 5x decimation by nearest
        # would drop 24 of every 25 pixels and alias thin features like rivers.
        # Class codes are the exception - averaging them invents classes.
        data = src.read(
            indices,
            out_shape=(len(indices), height, width),
            resampling=Resampling.nearest if categorical else Resampling.average,
        )
        nodata = src.nodata

    if categorical:
        lut = np.zeros((256, 3), dtype="uint8")
        for code, colour in CLASS_COLOURS.items():
            lut[code] = colour
        rgb = lut[data[0]].transpose(2, 0, 1)
        with MemoryFile() as memfile:
            profile = {"driver": "GTiff", "width": width, "height": height,
                       "count": 3, "dtype": "uint8"}
            with memfile.open(**profile) as tmp:
                tmp.write(rgb)
                rasterio.shutil.copy(tmp, out_path, driver="PNG")
        print(f"{out_path}  {width}x{height}  categorical, {len(np.unique(data))} classes present")
        return

    valid = np.ones(data.shape[1:], dtype=bool) if nodata is None else (data != nodata).any(axis=0)
    rgb = np.stack([stretch_to_byte(b, valid) for b in data])
    if rgb.shape[0] == 1:
        rgb = np.repeat(rgb, 3, axis=0)
    rgb[:, ~valid] = 0  # holes render black rather than as stretched noise

    profile = {
        "driver": "GTiff", "width": width, "height": height,
        "count": 3, "dtype": "uint8",
    }
    # The PNG driver is CreateCopy-only, so build the image in memory first.
    with MemoryFile() as memfile:
        with memfile.open(**profile) as tmp:
            tmp.write(rgb)
            rasterio.shutil.copy(tmp, out_path, driver="PNG")

    pct_valid = valid.mean() * 100
    print(f"{out_path}  {width}x{height}  bands {indices}  {pct_valid:.1f}% valid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raster", type=Path)
    parser.add_argument("--bands", nargs="+", help="band names or 1-based indices")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--categorical", action="store_true", help="class codes: fixed palette, no stretch"
    )
    args = parser.parse_args()

    out = args.out or args.raster.with_suffix(".quicklook.png")
    render(args.raster, args.bands, args.width, out, args.categorical)


if __name__ == "__main__":
    main()

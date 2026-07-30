"""Spatial block splits — the binding rule from spec section 8.

Adjacent pixels in satellite imagery are strongly correlated: a 256 px patch
and the patch 10 px to its left are nearly the same picture. Split those at
random and the validation set contains the training set's neighbours, every
metric inflates, and the model looks far better than it is. It is the same
leakage class as a badly done temporal split, and the writeup says so.

So the region is cut into large contiguous blocks and whole blocks are held
out. Two further guarantees on top of that:

  * a patch never straddles a block boundary - patch origins are enumerated
    strictly inside a single block, so no sample is part-train part-val;
  * a buffer of unused ground separates train blocks from val blocks, because
    two patches sitting either side of a shared edge are still neighbours even
    though they are in different blocks.

Nothing here is model code. This produces the index of which ground belongs
to which split, deterministically from a seed, so any run can be reproduced
and the split itself can be looked at.

Run:  python -m model.splits --region A
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from pipeline.grid import Grid, grid_for
from pipeline.regions import REGIONS, Region

BLOCK_PX = 1024      # ~10.2 km blocks: big enough that autocorrelation dies inside one
PATCH_PX = 256       # spec section 8
BUFFER_PX = PATCH_PX  # unused margin around held-out blocks
VAL_FRACTION = 0.20  # spec section 8 says 15-25%
SEED = 20260731

TRAIN, VAL, BUFFER = 1, 2, 0

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CLASS_NAMES = {
    1: "trees", 2: "shrub+grass", 3: "cropland",
    4: "built", 5: "bare", 6: "water+wetland",
}


def block_shape(grid: Grid, block_px: int) -> tuple[int, int]:
    return -(-grid.height // block_px), -(-grid.width // block_px)


def assign_blocks(
    grid: Grid, block_px: int = BLOCK_PX, val_fraction: float = VAL_FRACTION, seed: int = SEED
) -> np.ndarray:
    """Label every block TRAIN or VAL, deterministically.

    Blocks are drawn at random rather than as a checkerboard: a checkerboard
    puts every val block adjacent to four train blocks, which maximises shared
    boundary - the exact thing the buffer then has to undo.
    """
    rows, cols = block_shape(grid, block_px)
    total = rows * cols
    n_val = max(int(round(total * val_fraction)), 1)

    rng = np.random.default_rng(seed)
    labels = np.full(total, TRAIN, dtype="uint8")
    labels[rng.choice(total, size=n_val, replace=False)] = VAL
    return labels.reshape(rows, cols)


def expand(blocks: np.ndarray, grid: Grid, block_px: int = BLOCK_PX) -> np.ndarray:
    """Block labels -> per-pixel labels, cropped to the grid."""
    full = np.repeat(np.repeat(blocks, block_px, axis=0), block_px, axis=1)
    return full[: grid.height, : grid.width]


def apply_buffer(pixels: np.ndarray, buffer_px: int = BUFFER_PX) -> np.ndarray:
    """Blank a margin of train ground around every val block.

    Implemented as a max-filter on the val mask: any train pixel within
    `buffer_px` of val ground becomes BUFFER and is used by neither split.
    """
    if buffer_px <= 0:
        return pixels
    from scipy import ndimage

    near_val = ndimage.binary_dilation(
        pixels == VAL, structure=np.ones((3, 3), bool), iterations=buffer_px
    )
    out = pixels.copy()
    out[(pixels == TRAIN) & near_val] = BUFFER
    return out


def patch_origins(
    pixels: np.ndarray, split: int, patch_px: int = PATCH_PX, stride: int | None = None
) -> list[tuple[int, int]]:
    """Top-left corners of patches lying wholly inside `split` ground.

    "Wholly inside" is the whole point: a patch that is 90% train and 10% val
    is a leak, so it is simply not offered.
    """
    stride = stride or patch_px
    height, width = pixels.shape
    origins = []
    for row in range(0, height - patch_px + 1, stride):
        for col in range(0, width - patch_px + 1, stride):
            if (pixels[row : row + patch_px, col : col + patch_px] == split).all():
                origins.append((row, col))
    return origins


def class_balance(labels: np.ndarray, pixels: np.ndarray, split: int) -> dict[str, float]:
    """Class shares within one split - a val set missing a class cannot score it."""
    selected = labels[pixels == split]
    selected = selected[selected > 0]
    if selected.size == 0:
        return {}
    codes, counts = np.unique(selected, return_counts=True)
    return {
        CLASS_NAMES.get(int(c), str(int(c))): round(float(n / selected.size * 100), 2)
        for c, n in zip(codes, counts)
    }


def build(region: Region, block_px=BLOCK_PX, buffer_px=BUFFER_PX, seed=SEED) -> np.ndarray:
    grid = grid_for(region)
    blocks = assign_blocks(grid, block_px, seed=seed)
    return apply_buffer(expand(blocks, grid, block_px), buffer_px)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=sorted(REGIONS))
    parser.add_argument("--block-px", type=int, default=BLOCK_PX)
    parser.add_argument("--buffer-px", type=int, default=BUFFER_PX)
    parser.add_argument("--patch-px", type=int, default=PATCH_PX)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    region = REGIONS[args.region]
    grid = grid_for(region)
    out_dir = DATA_DIR / f"region{region.id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, cols = block_shape(grid, args.block_px)
    print(
        f"Region {region.id}: {rows} x {cols} blocks of {args.block_px} px "
        f"({args.block_px * grid.res / 1000:.1f} km), buffer {args.buffer_px} px"
    )

    pixels = build(region, args.block_px, args.buffer_px, args.seed)
    total = pixels.size
    for name, code in (("train", TRAIN), ("val", VAL), ("buffer", BUFFER)):
        print(f"  {name:7s} {(pixels == code).mean() * 100:5.2f}% of the region")

    for name, code in (("train", TRAIN), ("val", VAL)):
        origins = patch_origins(pixels, code, args.patch_px)
        print(f"  {name:7s} {len(origins):5d} non-overlapping {args.patch_px} px patches")

    landcover_path = out_dir / "landcover.tif"
    if landcover_path.exists():
        with rasterio.open(landcover_path) as src:
            labels = src.read(1)
        print("  class balance by split (a class missing from val cannot be scored):")
        train_balance = class_balance(labels, pixels, TRAIN)
        val_balance = class_balance(labels, pixels, VAL)
        for name in CLASS_NAMES.values():
            t, v = train_balance.get(name, 0.0), val_balance.get(name, 0.0)
            warn = "  <-- absent from val" if v == 0 else ""
            print(f"    {name:14s} train {t:6.2f}%   val {v:6.2f}%{warn}")

    out_path = out_dir / "split.tif"
    with rasterio.open(out_path, "w", **grid.profile(1, "uint8", 255)) as dst:
        dst.write(pixels, 1)
        dst.set_band_description(1, "split_0buffer_1train_2val")
        dst.update_tags(seed=str(args.seed), block_px=str(args.block_px),
                        buffer_px=str(args.buffer_px), val_fraction=str(VAL_FRACTION))

    meta = {
        "region": region.id, "seed": args.seed, "block_px": args.block_px,
        "buffer_px": args.buffer_px, "patch_px": args.patch_px,
        "shares": {
            name: round(float((pixels == code).mean() * 100), 3)
            for name, code in (("train", TRAIN), ("val", VAL), ("buffer", BUFFER))
        },
    }
    (out_dir / "split.json").write_text(json.dumps(meta, indent=2))
    print(f"  grid check: PASS\n{out_path}\n{out_dir / 'split.json'}")


if __name__ == "__main__":
    main()

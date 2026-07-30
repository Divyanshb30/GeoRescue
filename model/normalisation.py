"""Frozen input normalisation statistics, computed on training ground only.

Spec section 8: "Normalization stats computed per region-A training set,
frozen, documented." Each of those three words is load-bearing.

**Training set, not the region.** Mean and standard deviation computed over
all pixels include the held-out blocks. That is a real leak - small, and
completely invisible in the metrics, because the model never sees a val label,
only a val-informed scaling of its inputs. It is the same family of mistake as
fitting a scaler before the train/test split, which is the single most common
leak in tabular ML. Stats here are computed strictly where `split.tif` says
TRAIN.

**Frozen.** Written to a committed JSON and read back at train and inference
time. Recomputing them per run would mean two runs are not comparable, and
inference on Region B must use *Region A's* numbers - normalising Region B by
its own statistics would silently erase part of the very domain shift the
Phase-4 chapter exists to measure.

**Documented.** The file records the region, the split seed, the pixel count
behind each number, and the whole-region figures for comparison, so the size
of the leak avoided is visible rather than asserted.

Run:  python -m model.normalisation --region A
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from model.splits import SEED, TRAIN, build
from pipeline.grid import assert_matches, grid_for
from pipeline.regions import REGIONS, Region

BANDS = ("blue", "green", "red", "nir")
FILL = 0  # composite nodata
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STATS_DIR = Path(__file__).resolve().parent / "stats"

# Robust percentiles alongside mean/std: surface reflectance has a long bright
# tail (roofs, sand, residual haze) and a 2-98 scaling is often steadier than
# z-scoring. Recorded so the choice can be made later without recomputing.
PERCENTILES = (1, 2, 50, 98, 99)


def band_stats(values: np.ndarray) -> dict:
    """Statistics for one band, in float64 so 27 million uint16s don't drift."""
    values = values.astype("float64")
    pct = np.percentile(values, PERCENTILES)
    return {
        "count": int(values.size),
        "mean": round(float(values.mean()), 3),
        "std": round(float(values.std()), 3),
        "min": float(values.min()),
        "max": float(values.max()),
        **{f"p{p}": round(float(v), 1) for p, v in zip(PERCENTILES, pct)},
    }


def compute(region: Region, seed: int = SEED) -> dict:
    grid = grid_for(region)
    composite = DATA_DIR / f"region{region.id}" / "s2_composite.tif"
    if not composite.exists():
        raise SystemExit(f"no composite at {composite} - run pipeline.composite first")

    split = build(region, seed=seed)
    train_mask = split == TRAIN

    train_stats, region_stats = {}, {}
    with rasterio.open(composite) as src:
        assert_matches(grid, src, composite.name)
        for index, name in enumerate(BANDS, start=1):
            band = src.read(index)
            real = band != FILL
            train_stats[name] = band_stats(band[real & train_mask])
            region_stats[name] = band_stats(band[real])
            print(
                f"  {name:6s} train mean {train_stats[name]['mean']:8.1f} "
                f"std {train_stats[name]['std']:7.1f}   "
                f"whole-region mean {region_stats[name]['mean']:8.1f} "
                f"std {region_stats[name]['std']:7.1f}"
            )

    return {
        "region": region.id,
        "source": "s2_composite.tif, bands B2/B3/B4/B8",
        "imagery_window": "2026-01-01/2026-04-30",
        "computed_on": "split == TRAIN only",
        "split_seed": seed,
        "train": train_stats,
        "whole_region_for_comparison": region_stats,
        "note": (
            "Use `train` for both training and inference, in every region. "
            "Normalising Region B by Region B's own statistics would erase "
            "part of the domain shift the Phase-4 analysis measures."
        ),
    }


def load(region_id: str) -> dict:
    """Frozen stats for a region, as written by this module."""
    path = STATS_DIR / f"norm_region{region_id}.json"
    if not path.exists():
        raise SystemExit(f"no frozen stats at {path} - run: python -m model.normalisation --region {region_id}")
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=sorted(REGIONS))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    region = REGIONS[args.region]
    print(f"Region {region.id}: normalisation stats over TRAIN blocks (seed {args.seed})")
    stats = compute(region, args.seed)

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STATS_DIR / f"norm_region{region.id}.json"
    out_path.write_text(json.dumps(stats, indent=2))

    print("\n  drift avoided by excluding held-out ground:")
    for name in BANDS:
        train, whole = stats["train"][name], stats["whole_region_for_comparison"][name]
        d_mean = abs(train["mean"] - whole["mean"]) / whole["std"] * 100
        print(
            f"    {name:6s} mean differs by {abs(train['mean'] - whole['mean']):6.2f} DN "
            f"({d_mean:.2f}% of a standard deviation)"
        )
    print(f"\n{out_path}")


if __name__ == "__main__":
    main()

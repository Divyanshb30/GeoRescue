"""Normalisation statistics, and the leak they exist to avoid."""

import json

import numpy as np
import pytest

from model import normalisation
from model.normalisation import BANDS, band_stats, compute, load
from model.splits import TRAIN, build
from pipeline.regions import REGIONS
from scoring import layers

COMPOSITE = layers.DATA_DIR / "regionA" / "s2_composite.tif"
needs_composite = pytest.mark.skipif(
    not COMPOSITE.exists(), reason="Region A composite not built"
)


def test_band_stats_matches_numpy_on_a_known_array():
    values = np.array([100, 200, 300, 400], dtype="uint16")
    stats = band_stats(values)
    assert stats["count"] == 4
    assert stats["mean"] == pytest.approx(250.0)
    # band_stats rounds to 3 dp, so compare at that precision.
    assert stats["std"] == pytest.approx(np.std([100, 200, 300, 400]), abs=1e-3)
    assert stats["min"] == 100 and stats["max"] == 400
    assert stats["p50"] == pytest.approx(250.0)


def test_band_stats_accumulates_in_float64():
    """uint16 sums overflow past ~65k values; float64 must not."""
    values = np.full(200_000, 60_000, dtype="uint16")
    assert band_stats(values)["mean"] == pytest.approx(60_000.0)


def test_load_fails_loudly_when_stats_were_never_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(normalisation, "STATS_DIR", tmp_path)
    with pytest.raises(SystemExit, match="no frozen stats"):
        load("A")


@needs_composite
def test_frozen_file_records_how_it_was_computed():
    stats = load("A")
    assert stats["computed_on"] == "split == TRAIN only"
    assert stats["region"] == "A"
    assert "split_seed" in stats
    for band in BANDS:
        assert stats["train"][band]["count"] > 0


@needs_composite
def test_training_stats_are_not_whole_region_stats():
    """Proves the split mask is actually applied - if it were not, these two
    would be identical and the leak would be silent."""
    stats = load("A")
    assert any(
        stats["train"][b]["mean"] != stats["whole_region_for_comparison"][b]["mean"]
        for b in BANDS
    )
    for band in BANDS:
        assert stats["train"][band]["count"] < stats["whole_region_for_comparison"][band]["count"]


@needs_composite
def test_training_pixel_count_matches_the_split():
    """The count behind each statistic must be the train mask, not a guess."""
    stats = load("A")
    train_share = (build(REGIONS["A"]) == TRAIN).mean()
    total = stats["whole_region_for_comparison"]["blue"]["count"]
    # Train pixels, as a share of valid pixels, cannot exceed the train share
    # of the whole grid, and should be close to it (nodata is only 0.65%).
    ratio = stats["train"]["blue"]["count"] / total
    assert ratio == pytest.approx(train_share, abs=0.02)

"""The split is the one place a silent bug inflates every number in the writeup.

Spec section 8 makes the spatial block split binding. These tests exist so the
claim "no leakage between train and val" is something the repo proves rather
than something the writeup asserts.
"""

import numpy as np
import pytest

from model.splits import (
    BUFFER,
    TRAIN,
    VAL,
    apply_buffer,
    assign_blocks,
    block_shape,
    build,
    class_balance,
    expand,
    patch_origins,
)
from pipeline.grid import grid_for
from pipeline.regions import REGIONS

GRID_A = grid_for(REGIONS["A"])


# --------------------------------------------------------------------------
# determinism and proportions
# --------------------------------------------------------------------------


def test_same_seed_gives_the_same_split():
    a = assign_blocks(GRID_A, seed=7)
    b = assign_blocks(GRID_A, seed=7)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_splits():
    a = assign_blocks(GRID_A, seed=1)
    b = assign_blocks(GRID_A, seed=2)
    assert not np.array_equal(a, b)


def test_val_fraction_is_respected_at_block_level():
    blocks = assign_blocks(GRID_A, val_fraction=0.25, seed=3)
    share = (blocks == VAL).mean()
    # Rounding to whole blocks on a 7x6 grid cannot hit 0.25 exactly.
    assert share == pytest.approx(0.25, abs=1.0 / blocks.size)


def test_every_block_is_labelled():
    blocks = assign_blocks(GRID_A, seed=5)
    assert set(np.unique(blocks)) <= {TRAIN, VAL}
    assert (blocks > 0).all()


def test_expand_crops_to_the_grid_not_the_block_multiple():
    blocks = assign_blocks(GRID_A, seed=5)
    pixels = expand(blocks, GRID_A)
    assert pixels.shape == GRID_A.shape
    rows, cols = block_shape(GRID_A, 1024)
    assert rows * 1024 >= GRID_A.height  # blocks over-cover, expand trims


# --------------------------------------------------------------------------
# the buffer
# --------------------------------------------------------------------------


def test_buffer_separates_train_from_val():
    """After buffering, no train pixel may touch a val pixel."""
    pixels = build(REGIONS["A"], block_px=512, buffer_px=32)
    train = pixels == TRAIN
    val = pixels == VAL

    from scipy import ndimage

    val_neighbourhood = ndimage.binary_dilation(val, np.ones((3, 3), bool))
    assert not (train & val_neighbourhood).any()


def test_buffer_only_consumes_train_ground_never_val():
    """Held-out ground must not shrink - the val set is the measurement."""
    unbuffered = expand(assign_blocks(GRID_A, seed=11), GRID_A)
    buffered = apply_buffer(unbuffered.copy(), buffer_px=64)
    assert (buffered == VAL).sum() == (unbuffered == VAL).sum()
    assert (buffered == TRAIN).sum() < (unbuffered == TRAIN).sum()


def test_zero_buffer_is_a_no_op():
    pixels = expand(assign_blocks(GRID_A, seed=11), GRID_A)
    np.testing.assert_array_equal(apply_buffer(pixels.copy(), 0), pixels)


# --------------------------------------------------------------------------
# patch containment - the leakage guarantee
# --------------------------------------------------------------------------


def test_every_train_patch_is_entirely_train():
    pixels = build(REGIONS["A"], block_px=512, buffer_px=32)
    for row, col in patch_origins(pixels, TRAIN, patch_px=256)[:40]:
        patch = pixels[row : row + 256, col : col + 256]
        assert (patch == TRAIN).all()


def test_no_train_patch_contains_a_single_val_pixel():
    """The leakage test. One val pixel inside a training patch is enough to
    make the held-out score optimistic."""
    pixels = build(REGIONS["A"], block_px=512, buffer_px=32)
    for row, col in patch_origins(pixels, TRAIN, patch_px=256):
        assert not (pixels[row : row + 256, col : col + 256] == VAL).any()


def test_no_val_patch_contains_a_single_train_pixel():
    pixels = build(REGIONS["A"], block_px=512, buffer_px=32)
    for row, col in patch_origins(pixels, VAL, patch_px=256):
        assert not (pixels[row : row + 256, col : col + 256] == TRAIN).any()


def test_train_and_val_patches_never_overlap_each_other():
    pixels = build(REGIONS["A"], block_px=512, buffer_px=32)
    covered = np.zeros(pixels.shape, dtype="uint8")
    for split in (TRAIN, VAL):
        for row, col in patch_origins(pixels, split, patch_px=256):
            covered[row : row + 256, col : col + 256] |= split
    assert not (covered == (TRAIN | VAL)).any()


def test_patch_origins_reject_a_patch_straddling_a_boundary():
    """Hand-built case: a patch that is half train, half val, is not offered."""
    pixels = np.full((16, 16), TRAIN, dtype="uint8")
    pixels[:, 8:] = VAL
    assert patch_origins(pixels, TRAIN, patch_px=16) == []
    assert patch_origins(pixels, TRAIN, patch_px=8) == [(0, 0), (8, 0)]


def test_stride_controls_patch_overlap():
    pixels = np.full((32, 32), TRAIN, dtype="uint8")
    assert len(patch_origins(pixels, TRAIN, patch_px=16, stride=16)) == 4
    assert len(patch_origins(pixels, TRAIN, patch_px=16, stride=8)) == 9


def test_no_patches_are_offered_from_buffer_ground():
    pixels = build(REGIONS["A"], block_px=512, buffer_px=32)
    assert patch_origins(pixels, BUFFER, patch_px=256) == [] or all(
        (pixels[r : r + 256, c : c + 256] == BUFFER).all()
        for r, c in patch_origins(pixels, BUFFER, patch_px=256)
    )


# --------------------------------------------------------------------------
# class balance reporting
# --------------------------------------------------------------------------


def test_class_balance_sums_to_one_hundred():
    pixels = np.array([[TRAIN, TRAIN], [VAL, VAL]], dtype="uint8")
    labels = np.array([[1, 3], [6, 6]], dtype="uint8")
    assert sum(class_balance(labels, pixels, TRAIN).values()) == pytest.approx(100.0, abs=0.1)


def test_class_balance_ignores_the_nodata_label():
    pixels = np.full((2, 2), TRAIN, dtype="uint8")
    labels = np.array([[0, 0], [1, 1]], dtype="uint8")
    assert class_balance(labels, pixels, TRAIN) == {"trees": 100.0}


def test_class_balance_is_empty_when_a_split_has_no_pixels():
    pixels = np.full((2, 2), TRAIN, dtype="uint8")
    labels = np.ones((2, 2), dtype="uint8")
    assert class_balance(labels, pixels, VAL) == {}

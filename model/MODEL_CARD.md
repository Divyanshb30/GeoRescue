# Model card — GeoRescue land-cover segmentation

Status: **not yet trained.** Every section that depends on a training run is
marked `TBD — Phase 3`. Everything else is already determined by the data and
the split protocol, and is filled in with measured values rather than
placeholders, so that what the model is being asked to do is fixed *before*
the first run rather than described after it.

---

## Intended use

**In scope.** Producing a 6-class land-cover raster over the two precomputed
regions, as one of five inputs to a shelter-siting suitability score. Output
is decision support for assessment, not certified site planning.

**Out of scope.** Any region outside the two defined bboxes. Any date outside
the Jan–Apr 2026 imagery window without re-running the composite. Standalone
land-cover mapping presented as an authoritative product — WorldCover is the
authoritative product here, and this model is measured against it.

**Not to be used for** legal land classification, property assessment, or as
the sole basis for siting decisions affecting people.

---

## Task and classes

Semantic segmentation, Sentinel-2 → 6 classes merged from WorldCover's 11
(#011). Class 0 is ignore/nodata, skipped by the loss and excluded by the
scorer.

| code | class | Region A share | Region B share |
|---|---|---|---|
| 1 | trees | 70.05 % | 28.91 % |
| 2 | shrub + grassland | 7.00 % | 4.40 % |
| 3 | cropland | 16.07 % | 48.91 % |
| 4 | built-up | 5.21 % | 1.53 % |
| 5 | bare / sparse | 1.24 % | 1.11 % |
| 6 | water + wetland | 0.43 % | 15.15 % |

Built-up and water are the classes that drive scoring exclusions, and they are
among the rarest. **Trees:water in Region A is 163:1.**

---

## Data

| | |
|---|---|
| Inputs | Sentinel-2 L2A bands B2, B3, B4, B8 at 10 m |
| Imagery | Per-pixel median composite, 2026-01-01 → 2026-04-30, SCL-masked (#009) |
| Region A composite quality | median 12 clear looks/pixel, 0.652 % pixels with none |
| Region B composite quality | median ~24 clear looks/pixel, ~0 % empty |
| Labels | ESA WorldCover 2021 v200 — **weak supervision**, 5 years older than the imagery |
| Patch size | 256 × 256 |
| Normalisation | Frozen, `model/stats/norm_regionA.json`, computed on **training blocks only** (#016) |

Region A training statistics (DN): blue 1464.2 ± 299.3 · green 1671.3 ± 326.7 ·
red 1672.2 ± 415.2 · NIR 3108.8 ± 527.0. **Region A's statistics are used for
Region B as well** — normalising B by its own distribution would erase part of
the shift being measured.

---

## Split protocol

Spatial block split (#015), binding per spec §8. 1024 px (~10.2 km) blocks,
randomly assigned with seed 20260731, 20 % held out, and a 256 px buffer of
unused ground around every held-out block.

| | Region A | Region B |
|---|---|---|
| train / val / buffer | 66.1 / 17.9 / 16.1 % | 66.9 / 18.1 / 15.0 % |
| patches (stride 256) | 397 / 104 | 380 / 100 |

No patch spans a split boundary — enforced by construction and brute-force
tested in `tests/test_splits.py`. All six classes appear in both splits of
both regions.

---

## Architecture

Plain U-Net, ~4 depth levels, written by the project owner. **No pretrained
encoder in v1** — a deliberate cost of a few mIoU points, paid for the ability
to explain every layer (spec §12, §14.1).

`TBD — Phase 3:` exact channel widths, parameter count, normalisation layers.

---

## Training

`TBD — Phase 3:` loss weights, optimiser, schedule, epochs, augmentation,
early-stopping criterion, wall-clock, hardware.

Committed in advance: class-weighted cross-entropy + Dice, with
median-frequency balancing rather than inverse-frequency — at 163:1 the latter
makes water gradients violent (§4.5).

---

## Evaluation

`TBD — Phase 3:` mIoU, per-class F1, confusion matrix on held-out blocks.

Committed in advance:

- **Pixel accuracy will not be reported as a headline.** A constant "trees"
  predictor scores 70 % on Region A. Per-class F1 and mIoU are the metrics.
- **Baseline to beat:** agreement-with-WorldCover on holdout blocks is the
  *floor*, not the target.
- **Gate 3 bar:** the model must beat "just use WorldCover" on 2026 imagery in
  at least one demonstrable way — detecting post-2021 built-up change is the
  natural candidate — or the writeup states plainly why WorldCover remains the
  production layer.

---

## Cross-region generalisation (Phase 4)

Train on Region A, evaluate on Region B with no fine-tuning. `TBD` for the
numbers, but the *method* is fixed now, because it determines what the numbers
are allowed to mean:

A single headline "mIoU dropped from X to Y" explains nothing. Three confounds
are separated first (#014):

1. **Prior shift.** The label distribution moves violently — water 0.43 % →
   15.15 %, trees 70 % → 29 %. A model degrades on this alone, independent of
   whether Brahmaputra water *looks* like Ganga water.
2. **Imagery-depth confound.** Region A's eastern ~10 % has 4–7 clear looks
   against ~14 in the tile-overlap centre (#009). Structured error along that
   edge must be correlated against `s2_clear_count.tif` before it is blamed on
   the biome.
3. **Label-vintage drift.** WorldCover 2021 vs 2026 imagery, and the two
   regions have not changed at the same rate.

Only what survives all three earns the "cross-biome shift" claim.

---

## Known limitations

- **Labels are a model's output, not ground truth.** WorldCover 2021 has its
  own error profile; this model inherits it and cannot exceed it on
  agreement-based metrics by definition.
- **Five-year label drift** over a fast-growing Dehradun. Some "errors" are
  real new construction.
- **The composite is synthetic** — no pixel comes from a single date, so
  phenology is smeared across Jan–Apr.
- **Slope, a downstream input, comes from a DSM** — canopy and roofs inflate
  it, and land cover is least reliable at exactly those boundaries, so the two
  0.30-weight scoring factors fail together (#010).
- **Region A's eastern strip is thinner imagery** (#009).
- Trained and evaluated on two Indian regions only. Nothing here supports a
  claim about performance anywhere else.

---

## Reproducibility

| | |
|---|---|
| Scene selection | Pinned, `pipeline/manifests/region{A,B}_scenes.json` |
| OSM | Dated Geofabrik zone extract, not a live query (#013) |
| Split | Deterministic from seed 20260731, written to `split.tif` + `split.json` |
| Normalisation | Frozen JSON, committed |
| Training config | `TBD — Phase 3:` hashed config per run directory |

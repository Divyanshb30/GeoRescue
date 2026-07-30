"""Suitability scoring — the section 7 weighted overlay.

    OWNER-AUTHORED (spec section 14.1). This file is deliberately a stub.
    The scoring function is one of the four artifacts the owner writes
    personally; an agent may review it, never author it.

Everything around this module is finished and waiting: `scoring/layers.py`
delivers five aligned, validated arrays, and `scoring/site_extraction.py`
turns whatever this returns into ranked candidate sites. Filling in `score()`
is the last link to Gate 1.

--------------------------------------------------------------------------
THE CONTRACT
--------------------------------------------------------------------------
`score(stack)` takes a `LayerStack` and returns a `uint8` array of the same
shape, where:

    0        = excluded (a hard rule fired; the pixel is not a candidate)
    1..100   = suitability, higher is better
    0 also   = invalid (use `stack.valid`; do not score nodata pixels)

--------------------------------------------------------------------------
WHAT THE STACK GIVES YOU  (all same shape, all aligned, no checks needed)
--------------------------------------------------------------------------
    stack.slope       float32, degrees          0 .. ~66 in Region A
    stack.landcover   uint8, class codes        1 trees · 2 shrub+grass
                                                3 crop  · 4 built
                                                5 bare  · 6 water+wetland
                                                0 = ignore/nodata
    stack.elevation   float32, metres           251 .. 2766 in Region A
    stack.dist_road   float32, metres to nearest drivable road
    stack.dist_water  float32, metres to nearest perennial water
    stack.valid       bool, False where any layer has nodata

--------------------------------------------------------------------------
THE RULES YOU ARE IMPLEMENTING  (spec section 7, weights sum to 1.00)
--------------------------------------------------------------------------
    Slope        0.30   <=5 deg ideal; 5-10 penalised linearly; >15 EXCLUDED
    Land cover   0.30   grass/shrub/bare ideal; crop penalised; trees heavy
                        penalty; built/water/wetland EXCLUDED
    Flood proxy  0.20   elevation delta above nearest water pixel; low-lying
                        (<5 m delta within 1 km of water) heavily penalised
    Road access  0.10   distance decay; >5 km penalised
    Water supply 0.10   sweet-spot band: near but not flood-exposed

--------------------------------------------------------------------------
THINGS WORTH DECIDING BEFORE YOU WRITE  (each is a DECISIONS.md entry)
--------------------------------------------------------------------------
1. Exclusions vs weights. An excluded pixel is 0 regardless of how good its
   other factors are - so exclusions are applied *after* the weighted sum,
   not as a zero term inside it. Decide and write down why.
2. Normalisation shape per factor. Linear ramp, or something with a plateau?
   A hard 5-degree cliff makes 5.01 deg and 14.9 deg equally "penalised",
   which is not what the rule means.
3. The flood proxy needs BOTH elevation and dist_water: "height above the
   nearest water" is not a layer you have - it has to be derived. The cheap
   version is elevation minus the elevation at the nearest water pixel, which
   needs the EDT's index output, not just its distance.
4. The DSM caveat (DECISIONS #010). Slope near tree and building edges is
   inflated by canopy and roofs, and land cover is wrong at those same edges -
   the two 0.30-weight factors fail together. Decide whether the scorer
   compensates or whether site extraction handles it by preferring interiors.
5. Thresholds are v1 defaults pending a standards check against UNHCR site
   planning / Sphere (spec section 7, binding). Until sourced, they must not
   be presented as certified.

Run, once implemented:  python -m scoring.analyze --region A
"""

import numpy as np

from scoring.layers import LayerStack

EXCLUDED = 0

WEIGHTS = {
    "slope": 0.30,
    "landcover": 0.30,
    "flood": 0.20,
    "road": 0.10,
    "water": 0.10,
}


def score(stack: LayerStack) -> np.ndarray:
    """Suitability 1-100 per pixel, 0 where excluded or invalid.

    See the module docstring for the contract and the section 7 rules.
    """
    raise NotImplementedError(
        "scoring/suitability.py is owner-authored (spec section 14.1). "
        "See the module docstring for the contract, the available layers, "
        "and the five decisions to make before writing."
    )

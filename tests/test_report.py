"""The deterministic report, and the verifier that will police its successor."""

import json

import pytest

from reports.template import (
    _normalise,
    collect_input_numerals,
    numerals,
    render,
    verdict_for,
    verify,
)

PAYLOAD = {
    "region": "A",
    "imagery_window": "2026-01-01/2026-04-30",
    "landcover_source": "ESA WorldCover 2021 v200",
    "dem_source": "Copernicus GLO-30",
    "caveats": ["Composite imagery, Jan-Apr 2026 - not a single date."],
    "sites": [
        {
            "rank": 1, "lat": 30.25, "lon": 78.01, "score": 82.4, "score_max": 95.0,
            "area_ha": 3.75, "mean_slope_deg": 2.1, "max_slope_deg": 6.8,
            "dist_road_m": 340, "dist_water_m": 910, "mean_elevation_m": 655.0,
            "dominant_landcover": "shrub+grass", "flags": [],
        },
        {
            "rank": 2, "lat": 30.3, "lon": 78.05, "score": 41.0, "score_max": 60.0,
            "area_ha": 0.75, "mean_slope_deg": 8.4, "max_slope_deg": 13.2,
            "dist_road_m": 6100, "dist_water_m": 320, "mean_elevation_m": 720.0,
            "dominant_landcover": "cropland",
            "flags": ["over 5 km from the nearest drivable road", "small site: under 1 ha"],
        },
    ],
}


# --------------------------------------------------------------------------
# the zero-hallucination property
# --------------------------------------------------------------------------


def test_no_number_in_the_report_is_absent_from_the_input():
    """Spec §9's v1.5 check, applied to v1. A template cannot fail it - which
    is exactly why running it here proves the *verifier* works before it has
    to catch a real fabrication."""
    assert verify(render(PAYLOAD), PAYLOAD) == []


def test_verifier_catches_an_invented_figure():
    doctored = render(PAYLOAD) + "\n\nCapacity is approximately 1250 people."
    assert "1250" in verify(doctored, PAYLOAD)


def test_verifier_ignores_numbers_that_do_appear_in_the_input():
    honest = render(PAYLOAD) + "\n\nThe leading site covers 3.75 ha."
    assert verify(honest, PAYLOAD) == []


def test_verifier_walks_nested_structures():
    found = collect_input_numerals({"a": [{"b": 5}, "text 12"], "c": {"d": 7.5}})
    assert {"5", "12", "7.5"} <= found


def test_verifier_treats_booleans_as_not_numbers():
    """True must not smuggle in a 1."""
    assert collect_input_numerals({"flag": True}) == set()


@pytest.mark.parametrize("a,b", [(1, "1.0"), (2.50, "2.5"), (3.0, 3)])
def test_number_formatting_does_not_create_false_positives(a, b):
    assert _normalise(a) == _normalise(b)


def test_numerals_extracts_from_prose():
    assert numerals("slope 2.1 deg, road 340 m") == {"2.1", "340"}


# --------------------------------------------------------------------------
# content
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score,word", [(95, "strong"), (70, "strong"), (55, "workable"), (31, "marginal"), (5, "poor")]
)
def test_verdict_thresholds(score, word):
    assert verdict_for(score) == word


def test_report_lists_every_site():
    text = render(PAYLOAD)
    assert "Site 1" in text and "Site 2" in text
    assert "shrub+grass" in text and "cropland" in text


def test_report_surfaces_site_flags_rather_than_hiding_them():
    text = render(PAYLOAD)
    assert "over 5 km from the nearest drivable road" in text
    assert "small site: under 1 ha" in text


def test_report_carries_provenance_and_caveats():
    text = render(PAYLOAD)
    assert "Copernicus GLO-30" in text
    assert "ESA WorldCover 2021 v200" in text
    assert "not a single date" in text


def test_report_states_it_is_not_certified_site_planning():
    """Spec §3 honest labelling: the footer is not optional."""
    text = render(PAYLOAD).lower()
    assert "not certified site planning" in text
    assert "decision support" in text


def test_empty_result_is_reported_as_a_result_not_an_error():
    empty = {**PAYLOAD, "sites": []}
    text = render(empty)
    assert "No candidate sites" in text
    assert "not a failure" in text
    assert verify(text, empty) == []

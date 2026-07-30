"""Deterministic report generator — v1, zero hallucination surface.

Spec section 9 allows an LLM to write this prose only at v1.5, and only
constrained to the stats JSON. v1 is a template: computed statistics slotted
into fixed sentences. Nothing is generated, so nothing can be invented.

The interesting property is that this is testable. Every numeral in the output
must already appear in the input JSON - a rule that means nothing for a
template, because a template cannot do otherwise, but which is exactly the
check v1.5 will need against an LLM. Building the verifier now, against a
generator that provably passes it, means the check itself is trustworthy
before it has to catch anything.

Run:  python -m reports.template data/regionA/sites.json
"""

import argparse
import json
import re
from pathlib import Path

# Any run of digits, with optional decimal part. Deliberately crude: the
# verifier should over-collect rather than miss a fabricated figure.
NUMERAL = re.compile(r"\d+(?:\.\d+)?")

# Words that carry a verdict. Kept explicit so the mapping from number to
# claim is auditable, rather than buried in an f-string.
VERDICTS = (
    (70, "strong"),
    (50, "workable"),
    (30, "marginal"),
    (0, "poor"),
)


def verdict_for(score: float) -> str:
    for floor, word in VERDICTS:
        if score >= floor:
            return word
    return "poor"


def numerals(text: str) -> set[str]:
    """Every number appearing in a string, normalised for comparison."""
    return {_normalise(m) for m in NUMERAL.findall(text)}


def _normalise(value) -> str:
    """1, 1.0 and 1.00 are the same number; compare them that way."""
    number = float(value)
    return str(int(number)) if number == int(number) else str(round(number, 6))


def collect_input_numerals(payload: dict) -> set[str]:
    """Every number anywhere in the stats JSON, including inside strings."""
    found = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(key)
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
        elif isinstance(node, bool):
            pass
        elif isinstance(node, (int, float)):
            found.add(_normalise(node))
        elif isinstance(node, str):
            found.update(_normalise(m) for m in NUMERAL.findall(node))

    walk(payload)
    return found


def verify(text: str, payload: dict) -> list[str]:
    """Numerals in the report that do not appear in the input. Empty is correct."""
    allowed = collect_input_numerals(payload)
    return sorted(numerals(text) - allowed)


def site_paragraph(site: dict) -> str:
    lines = [
        f"**Site {site['rank']} — {verdict_for(site['score'])}** "
        f"(score {site['score']}, {site['area_ha']} ha)",
        f"Located at {site['lat']}, {site['lon']}, on ground classified as "
        f"{site['dominant_landcover']} at a mean elevation of "
        f"{site['mean_elevation_m']} m.",
        f"Mean slope {site['mean_slope_deg']}° (maximum {site['max_slope_deg']}°). "
        f"Nearest drivable road {site['dist_road_m']} m; nearest perennial water "
        f"{site['dist_water_m']} m.",
    ]
    if site["flags"]:
        lines.append("Cautions: " + "; ".join(site["flags"]) + ".")
    return "\n".join(lines)


def render(payload: dict) -> str:
    sites = payload.get("sites", [])
    region = payload.get("region", "?")

    parts = [f"# Shelter siting assessment — Region {region}", ""]

    if not sites:
        parts += [
            "## Verdict",
            "",
            "No candidate sites met the minimum contiguous area and score thresholds "
            "in the area analysed. This is a result, not a failure: the exclusion "
            "rules removed all qualifying ground.",
            "",
        ]
    else:
        best = sites[0]
        parts += [
            "## Verdict",
            "",
            f"{len(sites)} candidate site(s) identified. The highest-ranked is "
            f"{verdict_for(best['score'])}, scoring {best['score']} over "
            f"{best['area_ha']} ha.",
            "",
            "## Ranked sites",
            "",
            "| Rank | Score | Area (ha) | Mean slope (deg) | Road (m) | Water (m) | Cover |",
            "|---|---|---|---|---|---|---|",
        ]
        for site in sites:
            parts.append(
                f"| {site['rank']} | {site['score']} | {site['area_ha']} | "
                f"{site['mean_slope_deg']} | {site['dist_road_m']} | "
                f"{site['dist_water_m']} | {site['dominant_landcover']} |"
            )
        parts.append("")
        parts.append("## Site detail")
        parts.append("")
        for site in sites:
            parts.append(site_paragraph(site))
            parts.append("")

    parts += ["## Data provenance", ""]
    for label, key in (
        ("Imagery window", "imagery_window"),
        ("Land cover", "landcover_source"),
        ("Elevation", "dem_source"),
    ):
        if key in payload:
            parts.append(f"- {label}: {payload[key]}")

    caveats = payload.get("caveats", [])
    if caveats:
        parts += ["", "## What these numbers do not mean", ""]
        parts += [f"- {c}" for c in caveats]

    parts += [
        "",
        "---",
        "",
        "Outputs are *candidate sites for assessment* — decision support, "
        "not certified site planning.",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats", type=Path, help="sites.json from scoring.analyze")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.stats.read_text())
    text = render(payload)

    invented = verify(text, payload)
    if invented:
        raise SystemExit(
            f"report contains numbers absent from the input stats: {invented}\n"
            "this is the zero-hallucination check failing and must never happen for v1"
        )

    out = args.out or args.stats.with_suffix(".report.md")
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n---\nevery numeral traced to the input stats\n{out}")


if __name__ == "__main__":
    main()

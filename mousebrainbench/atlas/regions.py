"""Validated lookups over atlas-neutral brain-region records."""

from __future__ import annotations

from mousebrainbench.schemas import BrainRegion


def index_regions(regions: list[BrainRegion]) -> dict[str, BrainRegion]:
    """Index unique acronyms for configuration and reporting."""

    result = {region.acronym: region for region in regions}
    if len(result) != len(regions):
        raise ValueError("region acronyms must be unique")
    return result


def select_regions(regions: list[BrainRegion], acronyms: list[str]) -> list[BrainRegion]:
    """Select regions in requested order and reject unknown acronyms."""

    index = index_regions(regions)
    missing = set(acronyms) - set(index)
    if missing:
        raise ValueError(f"unknown region acronyms: {sorted(missing)}")
    return [index[acronym] for acronym in acronyms]

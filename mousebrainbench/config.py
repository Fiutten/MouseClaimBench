"""Strict configuration loading and object factories for Phase 1 experiments."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from mousebrainbench.connectivity.normalization import normalize_connectivity
from mousebrainbench.connectivity.synthetic import generate_synthetic_connectivity
from mousebrainbench.dynamics.linear_rate import LinearRateModel
from mousebrainbench.dynamics.wilson_cowan import WilsonCowanModel
from mousebrainbench.schemas import BrainRegion, ConnectivityMatrix
from mousebrainbench.simulation.perturbations import Perturbation, PerturbationType
from mousebrainbench.simulation.runner import SimulationConfig
from mousebrainbench.simulation.stimuli import RegionalStimulus

T = TypeVar("T")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and retain its source path for provenance."""

    source = Path(path).resolve()
    with source.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("experiment configuration must be a YAML mapping")
    config["_config_path"] = str(source)
    return config


def _construct(cls: type[T], values: dict[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def build_connectivity(config: dict[str, Any]) -> tuple[list[BrainRegion], ConnectivityMatrix]:
    spec = config["connectivity"]
    if spec.get("source") != "synthetic":
        raise ValueError("Phase 1 implements only explicitly synthetic connectivity")
    regions, connectivity = generate_synthetic_connectivity(
        int(spec["n_regions"]),
        density=float(spec.get("density", 0.1)),
        seed=int(spec.get("seed", 0)),
    )
    return regions, normalize_connectivity(connectivity, spec.get("normalization", "spectral_radius"))


def build_model(config: dict[str, Any]) -> LinearRateModel | WilsonCowanModel:
    spec = dict(config["model"])
    name = spec.pop("name")
    models = {"linear_rate": LinearRateModel, "wilson_cowan": WilsonCowanModel}
    if name not in models:
        raise ValueError(f"unsupported model: {name}")
    return _construct(models[name], spec)


def build_simulation_config(config: dict[str, Any]) -> SimulationConfig:
    return _construct(SimulationConfig, dict(config["simulation"]))


def _indices(acronyms: list[str], regions: list[BrainRegion]) -> tuple[int, ...]:
    index = {region.acronym: position for position, region in enumerate(regions)}
    missing = set(acronyms) - set(index)
    if missing:
        raise ValueError(f"unknown region acronyms: {sorted(missing)}")
    return tuple(index[acronym] for acronym in acronyms)


def build_stimuli(config: dict[str, Any], regions: list[BrainRegion]) -> tuple[RegionalStimulus, ...]:
    stimuli = []
    for raw in config.get("stimuli", []):
        spec = dict(raw)
        stimuli.append(
            RegionalStimulus(target_indices=_indices(spec.pop("targets"), regions), **spec)
        )
    return tuple(stimuli)


def build_perturbations(
    config: dict[str, Any], regions: list[BrainRegion]
) -> tuple[Perturbation, ...]:
    perturbations = []
    for raw in config.get("perturbations", []):
        spec = dict(raw)
        perturbations.append(
            Perturbation(
                kind=PerturbationType(spec.pop("kind")),
                target_indices=_indices(spec.pop("targets"), regions),
                **spec,
            )
        )
    return tuple(perturbations)

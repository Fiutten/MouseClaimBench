"""Numerical integration for MouseBrainBench dynamical models."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from mousebrainbench.dynamics.base import DynamicalModel
from mousebrainbench.schemas import ConnectivityMatrix, SimulationState
from mousebrainbench.simulation.perturbations import (
    Perturbation,
    enforce_state_constraints,
    perturb_inputs_and_connectivity,
)
from mousebrainbench.simulation.stimuli import RegionalStimulus, combined_stimulus


@dataclass(frozen=True)
class SimulationConfig:
    """Numerical settings independent of a particular dynamical model."""

    dt: float = 0.1
    t_max: float = 100.0
    seed: int = 0
    integrator: Literal["euler", "rk4"] = "rk4"
    noise_std: float = 0.0
    clip: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.dt <= 0 or self.t_max <= 0:
            raise ValueError("dt and t_max must be positive")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if self.clip is not None and self.clip[0] >= self.clip[1]:
            raise ValueError("clip lower bound must be less than upper bound")


def run_simulation(
    model: DynamicalModel,
    connectivity: ConnectivityMatrix,
    config: SimulationConfig,
    *,
    stimuli: tuple[RegionalStimulus, ...] = (),
    perturbations: tuple[Perturbation, ...] = (),
) -> SimulationState:
    """Integrate a square regional model with reproducible stochasticity."""

    if not connectivity.is_square:
        raise ValueError("simulation requires square aligned connectivity")
    n_regions = connectivity.n_regions
    weights = connectivity.dense_weights()
    rng = np.random.default_rng(config.seed)
    time = np.arange(0.0, config.t_max + config.dt * 0.5, config.dt)
    state = model.initial_state(n_regions, rng)
    state_history = np.empty((len(time), len(state)), dtype=float)
    state_history[0] = state
    started = perf_counter()

    def derivative(current_time: float, current_state: np.ndarray) -> tuple[np.ndarray, float]:
        stimulus = combined_stimulus(stimuli, current_time, n_regions)
        perturbed_state, perturbed_weights, perturbed_input, noise_scale = (
            perturb_inputs_and_connectivity(
                time=current_time,
                state=current_state,
                weights=weights,
                external_input=stimulus,
                noise_scale=config.noise_std,
                perturbations=perturbations,
                n_regions=n_regions,
            )
        )
        return model.derivative(perturbed_state, perturbed_weights, perturbed_input), noise_scale

    for index in range(1, len(time)):
        previous_time = time[index - 1]
        if config.integrator == "euler":
            slope, noise_scale = derivative(previous_time, state)
            next_state = state + config.dt * slope
        else:
            k1, noise_scale = derivative(previous_time, state)
            k2, _ = derivative(previous_time + config.dt / 2, state + config.dt * k1 / 2)
            k3, _ = derivative(previous_time + config.dt / 2, state + config.dt * k2 / 2)
            k4, _ = derivative(previous_time + config.dt, state + config.dt * k3)
            next_state = state + config.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if noise_scale:
            next_state += rng.normal(0.0, noise_scale * np.sqrt(config.dt), len(state))
        if config.clip is not None:
            next_state = np.clip(next_state, *config.clip)
        next_state = enforce_state_constraints(
            next_state,
            time=time[index],
            perturbations=perturbations,
            n_regions=n_regions,
        )
        if not np.all(np.isfinite(next_state)):
            raise FloatingPointError(f"non-finite state at simulation step {index}")
        state = next_state
        state_history[index] = state

    activity = np.vstack([model.activity(row, n_regions) for row in state_history])
    runtime = perf_counter() - started
    metadata = {
        "seed": config.seed,
        "dt": config.dt,
        "t_max": config.t_max,
        "integrator": config.integrator,
        "noise_std": config.noise_std,
        "runtime_seconds": runtime,
        "model": type(model).__name__,
        "connectivity_source": connectivity.metadata.get("source", "unknown"),
        "perturbations": [perturbation.as_dict() for perturbation in perturbations],
    }
    return SimulationState(
        time=time,
        activity=activity,
        region_ids=connectivity.source_region_ids,
        variables=model.state_variables(state_history, n_regions),
        metadata=metadata,
    )

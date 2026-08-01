"""Allen Visual Behavior Neuropixels integration using standard NWB/HDF5 fields."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd

from mousebrainbench.empirical import (
    EvokedRegionalProfile,
    EvokedRegionalTimecourse,
    RegionalActivity,
)

QUALITY_FILTER = {
    "quality": "good",
    "valid_data": True,
    "amplitude_cutoff_lt": 0.1,
    "presence_ratio_gt": 0.9,
    "isi_violations_lt": 0.5,
}


@dataclass(frozen=True)
class SessionQualification:
    """Pre-extraction decision derived solely from versioned project metadata."""

    session_id: int
    accepted: bool
    reason: str
    region_unit_counts: dict[str, int]
    mouse_id: int | None
    experience_level: str | None


class AllenVBNRepository:
    """Read-only access to a local Allen VBN release with explicit provenance."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.manifest_path = self.root.parent / "visual-behavior-neuropixels_project_manifest_v0.5.0.json"
        self.metadata_dir = self.root / "project_metadata"
        self.sessions_dir = self.root / "behavior_ecephys_sessions"
        if not self.manifest_path.exists() or not self.metadata_dir.exists():
            raise FileNotFoundError("Allen VBN manifest or project metadata is missing")

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text())

    def session_path(self, session_id: int) -> Path:
        path = self.sessions_dir / str(session_id) / f"ecephys_session_{session_id}.nwb"
        if not path.exists():
            raise FileNotFoundError(f"session NWB is not available locally: {session_id}")
        return path

    def available_session_ids(self) -> tuple[int, ...]:
        return tuple(sorted(int(path.parent.name) for path in self.sessions_dir.glob("*/*.nwb")))

    def verify_metadata_hashes(self) -> dict[str, bool]:
        """Verify the BLAKE2b hashes supplied by the Allen CloudCache manifest."""

        results = {}
        for name, record in self.manifest["metadata_files"].items():
            path = self.metadata_dir / f"{name}.csv"
            digest = hashlib.blake2b(path.read_bytes()).hexdigest() if path.exists() else ""
            results[name] = digest == record["file_hash"]
        return results

    def units(self) -> pd.DataFrame:
        return pd.read_csv(self.metadata_dir / "units.csv")

    def sessions(self) -> pd.DataFrame:
        return pd.read_csv(self.metadata_dir / "ecephys_sessions.csv")

    def qualified_units(self) -> pd.DataFrame:
        units = self.units()
        return units[
            (units["quality"] == QUALITY_FILTER["quality"])
            & (units["valid_data"] == QUALITY_FILTER["valid_data"])
            & (units["amplitude_cutoff"] < QUALITY_FILTER["amplitude_cutoff_lt"])
            & (units["presence_ratio"] > QUALITY_FILTER["presence_ratio_gt"])
            & (units["isi_violations"] < QUALITY_FILTER["isi_violations_lt"])
        ]

    def qualify_sessions(
        self,
        regions: tuple[str, ...],
        *,
        min_units_per_region: int,
        available_only: bool = True,
    ) -> tuple[SessionQualification, ...]:
        """Qualify available sessions without opening large NWB files."""

        available = set(self.available_session_ids())
        units = self.qualified_units()
        counts = units.groupby(["ecephys_session_id", "structure_acronym"]).size().unstack(fill_value=0)
        sessions = self.sessions().set_index("ecephys_session_id")
        decisions = []
        session_ids = available if available_only else set(int(value) for value in sessions.index)
        for session_id in sorted(session_ids):
            region_counts = {
                region: int(counts.at[session_id, region])
                if session_id in counts.index and region in counts.columns
                else 0
                for region in regions
            }
            missing = [region for region, count in region_counts.items() if count < min_units_per_region]
            row = sessions.loc[session_id]
            decisions.append(
                SessionQualification(
                    session_id=session_id,
                    accepted=not missing,
                    reason="accepted" if not missing else f"insufficient_units:{','.join(missing)}",
                    region_unit_counts=region_counts,
                    mouse_id=int(row["mouse_id"]) if pd.notna(row["mouse_id"]) else None,
                    experience_level=str(row["experience_level"])
                    if pd.notna(row["experience_level"])
                    else None,
                )
            )
        return tuple(decisions)

    def extract_spontaneous_activity(
        self,
        session_id: int,
        regions: tuple[str, ...],
        *,
        bin_size_seconds: float = 1.0,
        min_units_per_region: int = 20,
    ) -> RegionalActivity:
        """Aggregate quality-filtered unit spikes during the longest spontaneous block."""

        if bin_size_seconds <= 0:
            raise ValueError("bin_size_seconds must be positive")
        path = self.session_path(session_id)
        metadata = self.qualified_units()
        metadata = metadata[metadata["ecephys_session_id"] == session_id].set_index("unit_id")
        with h5py.File(path, "r") as nwb:
            unit_ids = np.asarray(nwb["units/id"], dtype=np.int64)
            missing_ids = set(unit_ids) - set(metadata.index)
            selected = metadata.reindex(unit_ids).dropna(subset=["structure_acronym"])
            selected = selected[selected["structure_acronym"].isin(regions)]
            starts = np.asarray(nwb["intervals/spontaneous_presentations/start_time"], dtype=float)
            stops = np.asarray(nwb["intervals/spontaneous_presentations/stop_time"], dtype=float)
            if len(starts) == 0:
                raise ValueError(f"session {session_id} has no spontaneous interval")
            interval_index = int(np.argmax(stops - starts))
            start, stop = float(starts[interval_index]), float(stops[interval_index])
            edges = np.arange(start, stop + bin_size_seconds, bin_size_seconds)
            if edges[-1] > stop:
                edges[-1] = stop
            if len(edges) < 3:
                raise ValueError("spontaneous interval is too short for requested bin size")
            spike_times = nwb["units/spike_times"]
            spike_index = np.asarray(nwb["units/spike_times_index"], dtype=np.int64)
            nwb_position = {int(unit_id): index for index, unit_id in enumerate(unit_ids)}
            activity = np.zeros((len(edges) - 1, len(regions)), dtype=float)
            unit_counts = []
            for region_index, region in enumerate(regions):
                region_ids = [int(value) for value in selected[selected["structure_acronym"] == region].index]
                if len(region_ids) < min_units_per_region:
                    raise ValueError(
                        f"session {session_id} has {len(region_ids)} qualified units in {region}; "
                        f"minimum is {min_units_per_region}"
                    )
                for unit_id in region_ids:
                    position = nwb_position[unit_id]
                    left = 0 if position == 0 else int(spike_index[position - 1])
                    right = int(spike_index[position])
                    activity[:, region_index] += np.histogram(spike_times[left:right], bins=edges)[0]
                widths = np.diff(edges)
                activity[:, region_index] /= len(region_ids) * widths
                unit_counts.append(len(region_ids))
        return RegionalActivity(
            time=(edges[:-1] + edges[1:]) / 2 - start,
            activity_hz=activity,
            region_acronyms=regions,
            unit_counts=tuple(unit_counts),
            session_id=session_id,
            metadata={
                "source": "Allen Visual Behavior Neuropixels",
                "biological": True,
                "manifest_version": self.manifest["manifest_version"],
                "data_pipeline": self.manifest["data_pipeline"],
                "nwb_path": str(path),
                "spontaneous_window_seconds": [start, stop],
                "bin_size_seconds": bin_size_seconds,
                "quality_filter": QUALITY_FILTER,
                "n_nwb_units_not_quality_qualified": len(missing_ids),
            },
        )

    def extract_change_response_profile(
        self,
        session_id: int,
        regions: tuple[str, ...],
        *,
        min_units_per_region: int = 20,
        window_seconds: float = 0.25,
    ) -> EvokedRegionalProfile:
        """Aggregate baseline-corrected firing around active visual-change events."""

        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        path = self.session_path(session_id)
        metadata = self.qualified_units()
        metadata = metadata[metadata["ecephys_session_id"] == session_id].set_index("unit_id")
        with h5py.File(path, "r") as nwb:
            unit_ids = np.asarray(nwb["units/id"], dtype=np.int64)
            selected = metadata.reindex(unit_ids).dropna(subset=["structure_acronym"])
            selected = selected[selected["structure_acronym"].isin(regions)]
            presentation_names = [
                name
                for name in nwb["intervals"]
                if name.startswith("Natural_Images_Lum_Matched_set_ophys_")
                and name.endswith("_presentations")
            ]
            if len(presentation_names) != 1:
                raise ValueError(
                    f"session {session_id} has {len(presentation_names)} matching image tables"
                )
            presentations = nwb[f"intervals/{presentation_names[0]}"]
            event_mask = (
                (np.asarray(presentations["is_change"], dtype=float) == 1)
                & (np.asarray(presentations["omitted"], dtype=float) == 0)
                & np.asarray(presentations["active"], dtype=bool)
            )
            events = np.asarray(presentations["start_time"], dtype=float)[event_mask]
            if len(events) < 2:
                raise ValueError(f"session {session_id} has fewer than two visual-change events")
            spike_times = nwb["units/spike_times"]
            spike_index = np.asarray(nwb["units/spike_times_index"], dtype=np.int64)
            nwb_position = {int(unit_id): index for index, unit_id in enumerate(unit_ids)}
            full = np.zeros(len(regions), dtype=float)
            odd = np.zeros(len(regions), dtype=float)
            even = np.zeros(len(regions), dtype=float)
            unit_counts = []
            for region_index, region in enumerate(regions):
                region_ids = [int(value) for value in selected[selected["structure_acronym"] == region].index]
                if len(region_ids) < min_units_per_region:
                    raise ValueError(
                        f"session {session_id} has {len(region_ids)} qualified units in {region}; "
                        f"minimum is {min_units_per_region}"
                    )
                unit_counts.append(len(region_ids))
                for unit_id in region_ids:
                    position = nwb_position[unit_id]
                    left = 0 if position == 0 else int(spike_index[position - 1])
                    right = int(spike_index[position])
                    spikes = np.asarray(spike_times[left:right], dtype=float)
                    event_indices = np.searchsorted(spikes, events, side="left")
                    response_counts = np.searchsorted(
                        spikes, events + window_seconds, side="left"
                    ) - event_indices
                    baseline_counts = event_indices - np.searchsorted(
                        spikes, events - window_seconds, side="left"
                    )
                    response = np.asarray(response_counts - baseline_counts, dtype=float)
                    full[region_index] += response.mean()
                    odd[region_index] += response[::2].mean()
                    even[region_index] += response[1::2].mean()
                scale = len(region_ids) * window_seconds
                full[region_index] /= scale
                odd[region_index] /= scale
                even[region_index] /= scale
        return EvokedRegionalProfile(
            response_hz=full,
            odd_event_response_hz=odd,
            even_event_response_hz=even,
            region_acronyms=regions,
            unit_counts=tuple(unit_counts),
            session_id=session_id,
            event_count=len(events),
            metadata={
                "source": "Allen Visual Behavior Neuropixels",
                "biological": True,
                "manifest_version": self.manifest["manifest_version"],
                "event_definition": "active non-omitted visual changes",
                "presentation_table": presentation_names[0],
                "baseline_window_seconds": [-window_seconds, 0.0],
                "response_window_seconds": [0.0, window_seconds],
                "quality_filter": QUALITY_FILTER,
            },
        )

    def extract_change_response_timecourse(
        self,
        session_id: int,
        regions: tuple[str, ...],
        *,
        min_units_per_region: int = 20,
        start_seconds: float = -0.25,
        stop_seconds: float = 0.75,
        bin_size_seconds: float = 0.05,
        event_time_offset_seconds: float = 0.0,
    ) -> EvokedRegionalTimecourse:
        """Extract event-aligned regional firing around active visual changes."""

        if not start_seconds < 0 < stop_seconds or bin_size_seconds <= 0:
            raise ValueError("timecourse requires start < 0 < stop and positive bins")
        relative_edges = np.arange(start_seconds, stop_seconds + bin_size_seconds * 0.5, bin_size_seconds)
        if len(relative_edges) < 3 or not np.isclose(relative_edges[-1], stop_seconds):
            raise ValueError("timecourse range must be evenly divisible by bin size")
        path = self.session_path(session_id)
        metadata = self.qualified_units()
        metadata = metadata[metadata["ecephys_session_id"] == session_id].set_index("unit_id")
        with h5py.File(path, "r") as nwb:
            unit_ids = np.asarray(nwb["units/id"], dtype=np.int64)
            selected = metadata.reindex(unit_ids).dropna(subset=["structure_acronym"])
            selected = selected[selected["structure_acronym"].isin(regions)]
            presentations = self._visual_presentations(nwb, session_id)
            event_mask = (
                (np.asarray(presentations["is_change"], dtype=float) == 1)
                & (np.asarray(presentations["omitted"], dtype=float) == 0)
                & np.asarray(presentations["active"], dtype=bool)
            )
            events = np.asarray(presentations["start_time"], dtype=float)[event_mask]
            if event_time_offset_seconds:
                events = events + event_time_offset_seconds
            if len(events) < 2:
                raise ValueError(f"session {session_id} has fewer than two visual-change events")
            spike_times = nwb["units/spike_times"]
            spike_index = np.asarray(nwb["units/spike_times_index"], dtype=np.int64)
            nwb_position = {int(unit_id): index for index, unit_id in enumerate(unit_ids)}
            event_activity = np.zeros((len(events), len(relative_edges) - 1, len(regions)), dtype=float)
            unit_counts = []
            for region_index, region in enumerate(regions):
                region_ids = [int(value) for value in selected[selected["structure_acronym"] == region].index]
                if len(region_ids) < min_units_per_region:
                    raise ValueError(
                        f"session {session_id} has {len(region_ids)} qualified units in {region}; "
                        f"minimum is {min_units_per_region}"
                    )
                unit_counts.append(len(region_ids))
                for unit_id in region_ids:
                    position = nwb_position[unit_id]
                    left = 0 if position == 0 else int(spike_index[position - 1])
                    right = int(spike_index[position])
                    spikes = np.asarray(spike_times[left:right], dtype=float)
                    event_edges = events[:, None] + relative_edges[None, :]
                    edge_indices = np.searchsorted(spikes, event_edges, side="left")
                    event_activity[:, :, region_index] += np.diff(edge_indices, axis=1)
                event_activity[:, :, region_index] /= len(region_ids) * bin_size_seconds
        return EvokedRegionalTimecourse(
            time=(relative_edges[:-1] + relative_edges[1:]) / 2,
            activity_hz=event_activity.mean(axis=0),
            odd_event_activity_hz=event_activity[::2].mean(axis=0),
            even_event_activity_hz=event_activity[1::2].mean(axis=0),
            region_acronyms=regions,
            unit_counts=tuple(unit_counts),
            session_id=session_id,
            event_count=len(events),
            metadata={
                "source": "Allen Visual Behavior Neuropixels",
                "biological": True,
                "manifest_version": self.manifest["manifest_version"],
                "event_definition": "active non-omitted visual changes",
                "time_range_seconds": [start_seconds, stop_seconds],
                "bin_size_seconds": bin_size_seconds,
                "event_time_offset_seconds": event_time_offset_seconds,
                "quality_filter": QUALITY_FILTER,
            },
        )

    @staticmethod
    def _visual_presentations(nwb: h5py.File, session_id: int) -> h5py.Group:
        presentation_names = [
            name
            for name in nwb["intervals"]
            if name.startswith("Natural_Images_Lum_Matched_set_ophys_")
            and name.endswith("_presentations")
        ]
        if len(presentation_names) != 1:
            raise ValueError(f"session {session_id} has {len(presentation_names)} matching image tables")
        return nwb[f"intervals/{presentation_names[0]}"]


def qualification_records(decisions: Iterable[SessionQualification]) -> list[dict[str, Any]]:
    """Convert immutable decisions to JSON-safe audit records."""

    return [
        {
            "session_id": item.session_id,
            "accepted": item.accepted,
            "reason": item.reason,
            "region_unit_counts": item.region_unit_counts,
            "mouse_id": item.mouse_id,
            "experience_level": item.experience_level,
        }
        for item in decisions
    ]

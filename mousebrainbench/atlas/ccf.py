"""Allen CCF integration boundary.

No Allen loader is implemented in Phase 1. Keeping this boundary explicit
prevents synthetic region indices from being mistaken for CCF coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CCFProvenance:
    """Required provenance for a future Allen CCF-derived region set."""

    version: str
    resolution_um: int
    source_url: str
    downloaded_at: str

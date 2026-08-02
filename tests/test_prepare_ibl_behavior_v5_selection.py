import pandas as pd

from scripts.prepare_ibl_behavior_v5_selection import ROLE_SIZES, build_selection


def test_selection_is_mouse_disjoint_deterministic_and_role_complete() -> None:
    total = sum(ROLE_SIZES.values())
    fixture_rows = []
    cluster_rows = []
    for index in range(total + 2):
        subject = f"mouse-{index:03d}"
        for insertion in range(2):
            pid = f"pid-{index:03d}-{insertion}"
            fixture_rows.append(
                {
                    "pid": pid,
                    "eid": f"eid-{index:03d}-{insertion}",
                    "probe_name": f"probe0{insertion}",
                    "session_number": 1,
                    "date": "2025-01-01",
                    "subject": subject,
                    "lab": f"lab-{index % 4}",
                }
            )
            for unit in range(60):
                cluster_rows.append(
                    {
                        "pid": pid,
                        "label": 1,
                        "acronym": "Isocortex" if unit % 2 else "TH",
                    }
                )
    fixture = pd.DataFrame(fixture_rows)
    clusters = pd.DataFrame(cluster_rows)
    excluded = {"mouse-000", "mouse-001"}

    first = build_selection(fixture, clusters, excluded_subjects=excluded)
    second = build_selection(fixture.sample(frac=1.0), clusters, excluded_subjects=excluded)

    assert first == second
    assert len({row["subject"] for row in first}) == total
    assert not excluded & {row["subject"] for row in first}
    assert {role: sum(row["role"] == role for row in first) for role in ROLE_SIZES} == ROLE_SIZES

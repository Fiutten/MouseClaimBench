import numpy as np
import pandas as pd

from mousebrainbench.benchmarks.causalbench_transport import (
    _pair_feature_and_label,
    _strong_k562_genes,
    benjamini_hochberg,
    deterministic_fold_indices,
    intervention_effect_matrices,
)
from mousebrainbench.knowledge import load_default_profile


def test_cell_folds_are_deterministic_balanced_and_disjoint() -> None:
    cells = [f"cell-{index}" for index in range(99)]
    first = deterministic_fold_indices(cells, namespace="gene-a", maximum_per_fold=20)
    second = deterministic_fold_indices(cells, namespace="gene-a", maximum_per_fold=20)

    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))
    assert [len(fold) for fold in first] == [20, 20, 20]
    assert len(set(np.concatenate(first))) == 60


def test_benjamini_hochberg_is_monotone_in_ranked_p_values() -> None:
    values = np.asarray([0.03, 0.001, 0.04, 0.20, 0.009])
    adjusted = benjamini_hochberg(values)
    order = np.argsort(values)

    assert np.all(np.diff(adjusted[order]) >= 0.0)
    assert np.all(adjusted >= values)
    assert np.all(adjusted <= 1.0)


def test_strong_perturbation_filter_returns_h5ad_ensembl_identifier(tmp_path) -> None:
    path = tmp_path / "summary.xlsx"
    frame = pd.DataFrame(
        {
            "genetic perturbation": [
                "10023_ZC3H18_P1P2_ENSG00000158545",
                "10040_ZCCHC9_P1P2_ENSG00000131732",
            ],
            "Number of DEGs (anderson-darling)": [51, 50],
            "percent knockdown": [-0.31, -0.90],
            "number of cells (filtered)": [26, 100],
        }
    )
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name="TabB_K562_day6_summary_stat", index=False)

    assert _strong_k562_genes(path) == {"ENSG00000158545"}


def test_intervention_matrices_recover_a_directed_effect() -> None:
    rng = np.random.default_rng(42)
    genes = ("a", "b", "c")
    controls = tuple(rng.normal(size=(40, len(genes))) for _ in range(3))
    source_a = tuple(
        control + np.asarray([0.0, 2.0, 0.0])
        for control in tuple(rng.normal(size=(40, len(genes))) for _ in range(3))
    )
    source_b = tuple(rng.normal(size=(40, len(genes))) for _ in range(3))
    source_c = tuple(rng.normal(size=(40, len(genes))) for _ in range(3))
    matrices = intervention_effect_matrices(
        {"non-targeting": controls, "a": source_a, "b": source_b, "c": source_c},
        genes,
    )

    assert np.all(matrices["effect"][:, 0, 1] > 1.0)
    assert np.all(matrices["q_value"][:, 0, 1] < 0.05)
    assert np.all(matrices["q_value"][:, 0, 0] == 1.0)


def test_reference_fold_changes_labels_but_never_policy_features() -> None:
    claims = tuple(item.claim for item in load_default_profile().requirements)
    evidence_effect = np.asarray([[[0.0, 1.0], [0.0, 0.0]]] * 3, dtype=float)
    evidence_p = np.asarray([[[1.0, 0.001], [1.0, 1.0]]] * 3, dtype=float)
    evidence_q = evidence_p.copy()
    positive = {"effect": evidence_effect.copy(), "p_value": evidence_p, "q_value": evidence_q}
    negative = {key: value.copy() for key, value in positive.items()}
    negative["effect"][2, 0, 1] = 0.0
    negative["p_value"][2, 0, 1] = 1.0
    negative["q_value"][2, 0, 1] = 1.0

    positive_feature, positive_label, _ = _pair_feature_and_label(
        dataset="fixture",
        source="a",
        target="b",
        source_index=0,
        target_index=1,
        matrices=positive,
        sample_size=40,
        claim_names=claims,
    )
    negative_feature, negative_label, _ = _pair_feature_and_label(
        dataset="fixture",
        source="a",
        target="b",
        source_index=0,
        target_index=1,
        matrices=negative,
        sample_size=40,
        claim_names=claims,
    )

    assert np.array_equal(positive_feature, negative_feature)
    assert positive_label[claims.index("causal")] == 1
    assert negative_label[claims.index("causal")] == 0

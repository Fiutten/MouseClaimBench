import numpy as np

from mousebrainbench.validation.content_validity import (
    binary_fleiss_kappa,
    dimension_summary,
    item_cvi,
    modified_kappa,
)


def test_unanimous_valid_ratings_have_perfect_item_cvi() -> None:
    ratings = np.asarray([4, 4, 3, 4, 3, 4, 4])
    assert item_cvi(ratings) == 1.0
    assert modified_kappa(ratings) > 0.99


def test_binary_fleiss_kappa_distinguishes_agreement() -> None:
    agreed = np.asarray([[4] * 7, [1] * 7, [4] * 7, [1] * 7])
    assert binary_fleiss_kappa(agreed) == 1.0


def test_dimension_summary_preserves_item_level_failures() -> None:
    matrix = np.asarray([[4] * 7, [4, 4, 2, 2, 2, 2, 2]])
    result = dimension_summary(matrix, ("pass", "fail"))
    assert result["items"]["pass"]["item_cvi"] == 1.0
    assert result["items"]["fail"]["item_cvi"] < 0.78

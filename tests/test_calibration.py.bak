import numpy as np
import pytest

from src.evaluation.calibration import (
    brier,
    calibration_table,
    expected_calibration_error,
)


def test_well_calibrated_probabilities_have_low_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.random(20000) < p).astype(int)
    table = calibration_table(y, p, n_bins=10)
    assert expected_calibration_error(table) < 0.03


def test_overconfident_probabilities_have_high_ece():
    rng = np.random.default_rng(0)
    n = 20000
    p = np.where(rng.random(n) < 0.5, rng.uniform(0, 0.1, n), rng.uniform(0.9, 1, n))
    y = rng.integers(0, 2, n)  # sin relación con p
    table = calibration_table(y, p, n_bins=10)
    assert expected_calibration_error(table) > 0.3


def test_brier_of_perfect_predictions_is_zero():
    assert brier([0, 1], [0.0, 1.0]) == pytest.approx(0.0)
import pandas as pd
import pytest

from src.splitting.walk_forward import walk_forward_folds

MONTHLY = pd.date_range("2011-01-01", "2011-12-01", freq="MS")


def test_every_fold_respects_the_embargo():
    folds = walk_forward_folds(MONTHLY, embargo_days=90, min_train_snapshots=2)
    assert folds
    for train_end, test_date in folds:
        assert test_date - train_end >= pd.Timedelta(days=90)


def test_train_end_is_the_latest_allowed_snapshot():
    folds = walk_forward_folds(MONTHLY, embargo_days=90, min_train_snapshots=3)
    # 2011-06-01 - 90 d = 2011-03-03 -> el último snapshot permitido es 2011-03-01
    assert folds[0] == (pd.Timestamp("2011-03-01"), pd.Timestamp("2011-06-01"))
    # y ningún snapshot posterior a train_end también cumple el embargo
    for train_end, test_date in folds:
        later = MONTHLY[(MONTHLY > train_end) & (MONTHLY <= test_date - pd.Timedelta(days=90))]
        assert len(later) == 0


def test_each_test_snapshot_appears_once():
    folds = walk_forward_folds(MONTHLY, embargo_days=90, min_train_snapshots=2)
    tests = [t for _, t in folds]
    assert len(tests) == len(set(tests))


def test_min_train_snapshots_is_respected():
    folds = walk_forward_folds(MONTHLY, embargo_days=90, min_train_snapshots=4)
    for train_end, _ in folds:
        assert (MONTHLY <= train_end).sum() >= 4


def test_gap_of_exactly_the_embargo_is_allowed():
    dates = pd.DatetimeIndex(["2011-01-01", "2011-04-01"])  # exactamente 90 días
    assert walk_forward_folds(dates, embargo_days=90, min_train_snapshots=1) == [
        (pd.Timestamp("2011-01-01"), pd.Timestamp("2011-04-01"))
    ]


def test_returns_empty_when_not_enough_history():
    assert walk_forward_folds(MONTHLY, embargo_days=90, min_train_snapshots=50) == []


def test_unsorted_and_duplicated_dates_are_handled():
    shuffled = list(MONTHLY[::-1]) + [MONTHLY[0]]
    assert walk_forward_folds(shuffled, 90, 2) == walk_forward_folds(MONTHLY, 90, 2)


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        walk_forward_folds(MONTHLY, embargo_days=-1)
    with pytest.raises(ValueError):
        walk_forward_folds(MONTHLY, embargo_days=90, min_train_snapshots=0)

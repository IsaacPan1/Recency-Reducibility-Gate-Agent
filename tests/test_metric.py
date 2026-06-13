"""Tests for standardized_mean_shift."""

import math

import numpy as np
import pytest

from recency_cv import standardized_mean_shift


def test_identical_arrays_zero():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert standardized_mean_shift(a, a) == 0.0


def test_too_few_values_returns_nan():
    # left side has <5 valid values
    assert math.isnan(standardized_mean_shift([1, 2, 3, 4], [1, 2, 3, 4, 5, 6]))
    # right side has <5 valid values
    assert math.isnan(standardized_mean_shift([1, 2, 3, 4, 5], [1, 2, 3]))


def test_hand_computed_case():
    # a=[1..5] mean 3, b=[3..7] mean 5, |diff|=2.
    # population var of each is 2.0; pooled_std = sqrt((2+2)/2) = sqrt(2).
    # result = 2 / sqrt(2) = sqrt(2).
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [3.0, 4.0, 5.0, 6.0, 7.0]
    assert standardized_mean_shift(a, b) == pytest.approx(math.sqrt(2.0))


def test_drops_nans():
    # NaNs are dropped before any computation: a NaN-padded array must give the
    # exact same answer as the clean one.
    a_clean = [1.0, 2.0, 3.0, 4.0, 5.0]
    b_clean = [3.0, 4.0, 5.0, 6.0, 7.0]
    a_nan = [1.0, 2.0, np.nan, 3.0, 4.0, 5.0, np.nan]
    b_nan = [np.nan, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert standardized_mean_shift(a_nan, b_nan) == standardized_mean_shift(a_clean, b_clean)


def test_nan_drop_can_trigger_too_few():
    # After dropping NaNs the left side has only 4 valid values -> nan.
    a = [1.0, 2.0, 3.0, 4.0, np.nan]
    b = [3.0, 4.0, 5.0, 6.0, 7.0]
    assert math.isnan(standardized_mean_shift(a, b))

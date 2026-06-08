import numpy as np
import pandas as pd
import pytest
from rogii.preprocess import detect_ps, interpolate_gr


def test_detect_ps(hw):
    ps = detect_ps(hw)
    assert ps == 200  # fixture sets TVT_input NaN from index 200


def test_detect_ps_no_nan():
    df = pd.DataFrame({'TVT_input': [1.0, 2.0, 3.0]})
    assert detect_ps(df) == 3  # returns len(df)


def test_interpolate_gr_fills_nans(hw):
    result = interpolate_gr(hw)
    assert result['GR'].isna().sum() == 0


def test_interpolate_gr_imputed_flag(hw):
    orig_nan_count = hw['GR'].isna().sum()
    result = interpolate_gr(hw)
    assert result['gr_imputed'].sum() == orig_nan_count


def test_interpolate_gr_short_gap_linear():
    gr = pd.Series([1.0, np.nan, np.nan, 4.0])
    df = pd.DataFrame({'GR': gr})
    result = interpolate_gr(df)
    assert abs(result['GR'].iloc[1] - 2.0) < 0.1
    assert abs(result['GR'].iloc[2] - 3.0) < 0.1

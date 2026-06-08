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


from rogii.preprocess import calibrate_gr, extract_scalars

def test_calibrate_gr_reduces_mismatch(hw, tw):
    ps = 200
    a, b, hw_cal = calibrate_gr(hw, tw, ps)
    known = hw.iloc[:ps].dropna(subset=['TVT_input', 'GR'])
    tw_at_tvt = np.interp(known['TVT_input'].values, tw['TVT'].values, tw['GR'].values)
    err_before = np.mean((known['GR'].values - tw_at_tvt) ** 2)
    known_cal = hw_cal.iloc[:ps].dropna(subset=['TVT_input', 'GR'])
    err_after = np.mean((known_cal['GR'].values - tw_at_tvt) ** 2)
    assert err_after < err_before

def test_extract_scalars_keys(hw):
    s = extract_scalars(hw, 200)
    for key in ['last_known_tvt', 'slope_tvt_md_all', 'z_span', 'eval_zone_length']:
        assert key in s

def test_extract_scalars_positive_slope(hw):
    s = extract_scalars(hw, 200)
    assert s['slope_tvt_md_all'] > 0

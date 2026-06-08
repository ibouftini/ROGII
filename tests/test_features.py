import numpy as np
import pandas as pd
import pytest
from rogii.features import (
    build_alignment_df, compute_anchor_offsets,
    compute_formation_features, compute_b_well,
)

@pytest.fixture
def dummy_alignment(hw):
    n = len(hw) - 200
    base = np.linspace(12050.0, 12400.0, n)
    return {
        'pf_ancc': base, 'pf_z': base,
        'beam_ref': base, 'cons': base, 'sm5': base,
        'vcons': base, 'mod': base, 'loose': base, 'vloose': base, 'ultra': base,
        'sc8_tvt': base, 'sc15_tvt': base, 'sc25_tvt': base,
        'sc8_conf': 0.7, 'sc15_conf': 0.7, 'sc25_conf': 0.7, 'sc_trust': 0.3,
        'hyb_ref': base,
        'dtw_r20_mean': base, 'dtw_r20_std': np.zeros(n), 'dtw_r20_cv': np.zeros(n),
        'dtw_r50_mean': base, 'dtw_r50_std': np.zeros(n), 'dtw_r50_cv': np.zeros(n),
        'dtw_r100_mean': base, 'dtw_r100_std': np.zeros(n), 'dtw_r100_cv': np.zeros(n),
        'dtw_r200_mean': base, 'dtw_r200_std': np.zeros(n), 'dtw_r200_cv': np.zeros(n),
    }

def test_alignment_df_shape(hw, dummy_alignment):
    df = build_alignment_df(hw, ps_idx=200, alignment=dummy_alignment)
    assert len(df) == len(hw) - 200
    assert 'pf_ancc' in df.columns

def test_anchor_offsets_shape(hw, tw, dummy_alignment):
    gr = hw['GR'].ffill().values
    baseline = dummy_alignment['beam_ref']
    df = compute_anchor_offsets(baseline, tw['TVT'].values, tw['GR'].values, gr[200:])
    assert df.shape == (len(hw) - 200, 11)

def test_formation_features_columns(hw, tw):
    formations = {f: 12050.0 + i * 100 for i, f in
                  enumerate(['ANCC','ASTNU','ASTNL','EGFDU','EGFDL','BUDA'])}
    b_well = compute_b_well(hw, ps_idx=200, formation_depths=formations)
    df = compute_formation_features(hw, ps_idx=200, formation_depths=formations, b_well=b_well)
    for f in ['ANCC','ASTNU','ASTNL','EGFDU','EGFDL','BUDA']:
        assert f'tvt_fw_{f}' in df.columns
        assert f'form_rmse_{f}' in df.columns

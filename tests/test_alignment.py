import numpy as np
import pytest
from rogii.alignment.particle_filter import run_pf, run_pf_variants
from rogii.alignment.beam import run_beam_configs
import config


def test_run_pf_shape(hw, tw):
    n_eval = len(hw) - 200
    gr = hw['GR'].ffill().values[200:]
    traj = run_pf(gr, tw['TVT'].values, tw['GR'].values, tvt_start=12050.0, n_seeds=2)
    assert traj.shape == (n_eval,)


def test_run_pf_monotone(hw, tw):
    gr = hw['GR'].ffill().values[200:]
    traj = run_pf(gr, tw['TVT'].values, tw['GR'].values, tvt_start=12050.0, n_seeds=2)
    # TVT increments should be >= 0 on average
    assert np.diff(traj).mean() >= 0


def test_run_pf_variants_keys(hw, tw):
    gr = hw['GR'].ffill().values[200:]
    cfg = dict(n_particles=10, sigma_0=4.5, gr_scales=[5], n_seeds=2)
    result = run_pf_variants(gr, tw['TVT'].values, tw['GR'].values, 12050.0, cfg)
    assert 'pf_ancc' in result and 'pf_z' in result


def test_beam_configs_keys(hw, tw):
    gr = hw['GR'].ffill().values[200:]
    result = run_beam_configs(gr, tw['TVT'].values, tw['GR'].values, 12050.0, config.BEAM_CONFIGS)
    for key in ['cons', 'sm5', 'beam_ref']:
        assert key in result


def test_beam_ref_is_average(hw, tw):
    gr = hw['GR'].ffill().values[200:]
    result = run_beam_configs(gr, tw['TVT'].values, tw['GR'].values, 12050.0, config.BEAM_CONFIGS)
    np.testing.assert_allclose(result['beam_ref'], (result['cons'] + result['sm5']) / 2.0)


def test_beam_output_shape(hw, tw):
    n_eval = len(hw) - 200
    gr = hw['GR'].ffill().values[200:]
    result = run_beam_configs(gr, tw['TVT'].values, tw['GR'].values, 12050.0, config.BEAM_CONFIGS)
    assert result['beam_ref'].shape == (n_eval,)


import pytest
from rogii.alignment.ncc import run_ncc_multiscale, compute_sc_trust

def test_sc_trust_clip():
    assert compute_sc_trust(0) == 0.0
    assert compute_sc_trust(200) == pytest.approx(0.6)
    assert compute_sc_trust(1000) == pytest.approx(0.6)  # clamped

def test_ncc_multiscale_keys(hw, tw):
    gr = hw['GR'].ffill().values
    baseline = np.linspace(12050.0, 12400.0, len(hw) - 200)
    result = run_ncc_multiscale(
        gr, tw['TVT'].values, tw['GR'].values, baseline, known_rows=200
    )
    for key in ['sc8_tvt', 'sc15_tvt', 'sc25_tvt', 'sc_trust', 'hyb_ref']:
        assert key in result

def test_ncc_hyb_ref_formula(hw, tw):
    gr = hw['GR'].ffill().values
    baseline = np.linspace(12050.0, 12400.0, len(hw) - 200)
    result = run_ncc_multiscale(gr, tw['TVT'].values, tw['GR'].values, baseline, known_rows=200)
    sc_trust = result['sc_trust']
    expected = (1 - sc_trust) * baseline + sc_trust * result['sc15_tvt']
    np.testing.assert_allclose(result['hyb_ref'], expected)


from rogii.alignment.dtw import run_dtw_all_radii

def test_dtw_output_keys(hw, tw):
    gr = hw['GR'].ffill().values[200:]
    result = run_dtw_all_radii(gr, tw['TVT'].values, tw['GR'].values, radii=[20, 50], k=3)
    for key in ['dtw_r20_mean', 'dtw_r20_std', 'dtw_r20_cv',
                'dtw_r50_mean', 'dtw_r50_std', 'dtw_r50_cv']:
        assert key in result

def test_dtw_std_positive(hw, tw):
    gr = hw['GR'].ffill().values[200:]
    result = run_dtw_all_radii(gr, tw['TVT'].values, tw['GR'].values, radii=[20], k=4)
    assert (result['dtw_r20_std'] >= 0).all()

def test_dtw_cv_definition(hw, tw):
    gr = hw['GR'].ffill().values[200:]
    result = run_dtw_all_radii(gr, tw['TVT'].values, tw['GR'].values, radii=[20], k=3)
    expected_cv = result['dtw_r20_std'] / (np.abs(result['dtw_r20_mean']) + 1e-6)
    np.testing.assert_allclose(result['dtw_r20_cv'], expected_cv)

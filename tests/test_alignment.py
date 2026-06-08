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

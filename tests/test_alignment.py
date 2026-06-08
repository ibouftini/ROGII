import numpy as np
import pytest
from rogii.alignment.particle_filter import run_pf, run_pf_variants


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

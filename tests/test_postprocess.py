import numpy as np
import pytest
from rogii.postprocess import apply_rampup, blend_pf, savgol_smooth, robust_polyfit, apply_uspace

def test_rampup_at_zero():
    d = np.ones(100)
    md = np.arange(100, dtype=float)
    out = apply_rampup(d, md, alpha=1.0, tau=85.0)
    assert out[0] < 0.02   # near zero at start

def test_rampup_converges():
    d = np.ones(100)
    md = np.arange(100, dtype=float)
    out = apply_rampup(d, md, alpha=1.0, tau=85.0)
    assert out[-1] > 0.68  # converged toward 1.0

def test_blend_pf_extremes():
    dm, dp = np.ones(10) * 2.0, np.ones(10) * 4.0
    np.testing.assert_allclose(blend_pf(dm, dp, w_pf=0.0), dm)
    np.testing.assert_allclose(blend_pf(dm, dp, w_pf=1.0), dp)

def test_savgol_smooth_shape():
    y = np.random.default_rng(0).normal(0, 1, 200)
    out = savgol_smooth(y)
    assert out.shape == y.shape

def test_robust_polyfit_recovers_poly():
    x = np.linspace(0, 1, 100)
    y = 2.0 * x ** 2 - x + 0.5
    fitted = robust_polyfit(x, y, degree=2)
    np.testing.assert_allclose(fitted, y, atol=0.01)

def test_uspace_smooth():
    z = np.linspace(-12100, -12200, 500)
    tvt = np.linspace(12050, 12400, 500) + np.random.default_rng(0).normal(0, 5, 500)
    out = apply_uspace(tvt, z, anchor_tvt=12050.0)
    # output should be smoother than input
    assert np.std(np.diff(out)) < np.std(np.diff(tvt))

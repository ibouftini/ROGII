# tests/conftest.py
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def tw() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 500
    tvt = np.linspace(12000.0, 12500.0, n)
    gr = 80.0 + 40.0 * np.sin(tvt / 20.0) + rng.normal(0, 5, n)
    geo = np.where(tvt < 12200, 'ASTNL', np.where(tvt < 12400, 'EGFDU', 'EGFDL'))
    return pd.DataFrame({'TVT': tvt, 'GR': gr, 'Geology': geo})

@pytest.fixture
def hw(tw) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 800
    tvt = np.linspace(12050.0, 12400.0, n)
    gr = np.interp(tvt, tw['TVT'].values, tw['GR'].values) + rng.normal(0, 8, n)
    null_idx = rng.choice(n, size=int(0.3 * n), replace=False)
    gr[null_idx] = np.nan
    tvt_input = tvt.copy()
    tvt_input[200:] = np.nan
    formations = {f: tvt + off for f, off in
                  zip(['ANCC','ASTNU','ASTNL','EGFDU','EGFDL','BUDA'],
                      [-500,-400,-200,0,100,200])}
    return pd.DataFrame({
        'MD': np.arange(10000.0, 10000.0 + n),
        'X': np.linspace(3e6, 3.01e6, n),
        'Y': np.full(n, 1.1e6),
        'Z': np.linspace(-12100.0, -12200.0, n),
        'GR': gr, 'TVT': tvt, 'TVT_input': tvt_input,
        **formations,
    })

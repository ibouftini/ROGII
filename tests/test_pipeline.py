import pytest
import numpy as np
import pandas as pd
from rogii.utils import load_hw, load_tw


def test_process_well_returns_welldata():
    from rogii.pipeline import process_well
    from rogii.neighbors import FormationPlaneKNN, FORMATIONS
    import config
    hw = load_hw('data/train/000d7d20__horizontal_well.csv')
    tw = load_tw('data/train/000d7d20__typewell.csv')
    knn = FormationPlaneKNN().fit(
        [(0, float(hw['X'].mean()), float(hw['Y'].mean()),
          {f: float(hw[f].mean()) if f in hw.columns else 0.0 for f in FORMATIONS})]
    )
    tw_index = {}
    wd = process_well('000d7d20', hw, tw, knn, tw_index, config)
    assert wd.ps_idx > 0
    assert len(wd.scalars) > 0


@pytest.mark.skip(reason="full pipeline test requires GPU/long compute; run manually")
def test_run_pipeline_creates_submission(tmp_path):
    import config
    from rogii.pipeline import run_pipeline
    sub = run_pipeline(config, mode='predict',
                       train_dir='data/train', test_dir='data/test',
                       models_dir=str(tmp_path))
    # no trained models in tmp_path, so result is empty but must be a DataFrame
    assert isinstance(sub, pd.DataFrame)
    assert 'tvt' in sub.columns or len(sub) == 0

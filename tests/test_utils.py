import pandas as pd
from rogii.utils import extract_wellname, load_hw, load_tw, list_wells, WellData


def test_extract_wellname():
    assert extract_wellname('data/train/abc12345__horizontal_well.csv') == 'abc12345'


def test_load_hw_columns():
    df = load_hw('data/train/000d7d20__horizontal_well.csv')
    assert 'GR' in df.columns and 'TVT_input' in df.columns
    assert 'WELLNAME' not in df.columns


def test_list_wells_pairs():
    pairs = list_wells('data/train')
    assert len(pairs) > 0
    hw_path, tw_path = pairs[0]
    assert '__horizontal_well' in hw_path and '__typewell' in tw_path
    assert extract_wellname(hw_path) == extract_wellname(tw_path)


def test_welldata_fields(hw, tw):
    wd = WellData(name='test', hw=hw, tw=tw, ps_idx=200,
                  scalars={}, formations={}, cluster_id=0)
    assert wd.name == 'test'
    assert wd.tw_match is None

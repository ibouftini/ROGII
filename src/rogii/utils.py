import os
import glob
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class WellData:
    name: str
    hw: pd.DataFrame
    tw: pd.DataFrame
    ps_idx: int
    scalars: dict
    formations: dict        # {form_name: float} imputed depths
    cluster_id: int
    tw_match: str | None = None
    a_cal: float = 1.0
    b_cal: float = 0.0


def extract_wellname(path: str) -> str:
    """8-char hash from path like .../abc12345__horizontal_well.csv"""
    return os.path.basename(path).split('__')[0]


def load_hw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_tw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def list_wells(data_dir: str) -> list[tuple[str, str]]:
    """Return sorted list of (hw_path, tw_path) pairs."""
    hw_paths = sorted(glob.glob(os.path.join(data_dir, '*__horizontal_well.csv')))
    tw_map = {extract_wellname(p): p
              for p in glob.glob(os.path.join(data_dir, '*__typewell.csv'))}
    return [(p, tw_map[extract_wellname(p)])
            for p in hw_paths if extract_wellname(p) in tw_map]

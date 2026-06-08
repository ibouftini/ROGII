import numpy as np
from rogii.neighbors import assign_cluster, FormationPlaneKNN

def test_assign_cluster_c0():
    assert assign_cluster(tw_gr_mean=85.0, y_coord=1.1e6) == 0

def test_assign_cluster_c1():
    assert assign_cluster(tw_gr_mean=115.0, y_coord=1.095e6) == 1

def test_assign_cluster_c2():
    assert assign_cluster(tw_gr_mean=115.0, y_coord=1.05e6) == 2

def test_knn_returns_six_formations():
    wells = [(0, 1.0, 2.0, {f: float(i) for i, f in
              enumerate(['ANCC','ASTNU','ASTNL','EGFDU','EGFDL','BUDA'])})
             for _ in range(15)]
    knn = FormationPlaneKNN(k=5).fit(wells)
    result = knn.predict(cluster_id=0, x=1.1, y=2.1)
    assert set(result.keys()) == {'ANCC','ASTNU','ASTNL','EGFDU','EGFDL','BUDA'}

def test_knn_same_cluster_only():
    # C0 wells at x=0, C1 wells at x=100
    wells_c0 = [(0, 0.0, 0.0, {f: 0.0 for f in ['ANCC','ASTNU','ASTNL','EGFDU','EGFDL','BUDA']})
                for _ in range(10)]
    wells_c1 = [(1, 100.0, 0.0, {f: 999.0 for f in ['ANCC','ASTNU','ASTNL','EGFDU','EGFDL','BUDA']})
                for _ in range(10)]
    knn = FormationPlaneKNN(k=5).fit(wells_c0 + wells_c1)
    result = knn.predict(cluster_id=0, x=1.0, y=0.0)
    # should not be influenced by C1 wells (999.0)
    assert result['ANCC'] < 100.0

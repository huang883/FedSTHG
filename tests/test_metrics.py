import pytest
from fedsthg.metrics import binary_metrics,fold_summary


def test_hand_calculated_metrics():
    m=binary_metrics([0,0,1,1],[.1,.6,.4,.8])
    assert m['ACC']==.5 and m['SEN']==.5 and m['SPE']==.5 and m['F1']==.5
    assert m['AUC']==.75
    assert (m['tn'],m['fp'],m['fn'],m['tp'])==(1,1,1,1)


def test_undefined_auc_and_sd_are_not_fabricated():
    m=binary_metrics([0,0],[.1,.2]);assert m['AUC'] is None and m['SEN'] is None
    assert fold_summary([m])['ACC']['sd'] is None
    with pytest.raises(ValueError):binary_metrics([0,1],[float('nan'),.3])

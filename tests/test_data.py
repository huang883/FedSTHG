import copy
import pytest
from fedsthg.data import synthetic_manifest, load_manifest, make_splits, audit_splits


def test_splits_disjoint_complete_deterministic_and_subject_level(tmp_path):
    path = synthetic_manifest(tmp_path)
    _, rows = load_manifest(path)
    a = make_splits(rows)
    assert a == make_splits(rows)
    assert a != make_splits(rows, seed=2)
    audit_splits(rows, a)
    corrupt = copy.deepcopy(a)
    c = corrupt[0]['clients']['virtual_0']
    c['test'].append(c['train'][0])
    with pytest.raises(ValueError, match='overlap'):
        audit_splits(rows, corrupt)


def test_checksum_failure(tmp_path):
    path = synthetic_manifest(tmp_path)
    _, rows = load_manifest(path)
    (tmp_path / rows[0]['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Checksum'):
        load_manifest(path)

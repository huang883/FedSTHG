import numpy as np
import pytest
from fedsthg.preprocessing import anchor_align, normalize_roi, fisher_fc, connectivity, read_roi


def test_anchor_identity_constant_and_resampling():
    x = np.arange(24.).reshape(8, 3)
    np.testing.assert_array_equal(anchor_align(x, 8), x)
    np.testing.assert_array_equal(anchor_align(np.ones((8, 3)), 15), np.ones((15, 3)))
    expected = np.stack([x[0:2].mean(0), x[2:6].mean(0), x[6:8].mean(0)])
    np.testing.assert_allclose(anchor_align(x, 3), expected)
    assert anchor_align(x, 15).shape == (15, 3)


def test_pearson_matches_independent_numpy():
    x = np.random.default_rng(7).normal(size=(80, 5))
    ref = np.arctanh(np.clip(np.corrcoef(x.T), -1 + 1e-6, 1 - 1e-6))
    np.fill_diagonal(ref, 0)
    np.testing.assert_allclose(fisher_fc(x), ref, atol=1e-6)
    np.testing.assert_allclose(normalize_roi(x).mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(normalize_roi(x).std(0), 1, atol=1e-6)


def test_constant_and_perfect_correlations():
    x = np.arange(10.)
    z = fisher_fc(np.stack([x, x, -x, np.ones(10)], axis=1))
    assert np.isfinite(z).all()
    assert z[0, 1] > 7 and z[0, 2] < -7
    assert (np.diag(z) == 0).all() and (z[3] == 0).all()


def test_windows_and_last_window():
    x = np.random.default_rng(3).normal(size=(17, 4))
    s, d = connectivity(x, 17, 6, 4)
    assert s.shape == (4, 4) and d.shape == (3, 4, 4)
    np.testing.assert_allclose(d[-1], fisher_fc(normalize_roi(x)[8:14]))
    np.testing.assert_allclose(d, d.transpose(0, 2, 1))


@pytest.mark.parametrize("x", [np.ones(3), np.ones((1, 3)), np.full((4, 3), np.nan)])
def test_invalid_input(x):
    with pytest.raises(ValueError):
        connectivity(x)


def test_invalid_window_and_reader(tmp_path):
    with pytest.raises(ValueError):
        connectivity(np.ones((10, 3)), 10, 11, 2)
    p = tmp_path / "roi.1D"
    p.write_text("# ROI headers\n1 2 3\n4 5 6\n")
    assert read_roi(p, 3).shape == (2, 3)
    with pytest.raises(ValueError):
        read_roi(p, 116)

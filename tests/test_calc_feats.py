import random
import pytest
import numpy as np
from eight_mile.utils import calc_nfeats


def test_use_nfeats():
    filtsz = [random.randint(1, 10) for _ in range(random.randint(2, 6))]
    input_nfeat = random.randint(1, 100)
    gold_nfeats = [input_nfeat] * len(filtsz)
    _, nfeat = calc_nfeats(filtsz, None, None, nfeats=input_nfeat)
    assert nfeat == gold_nfeats


def test_use_nfeats_filtsz_unchanged():
    gold_filtsz = [random.randint(1, 10) for _ in range(random.randint(2, 6))]
    nfeat = random.randint(1, 100)
    filtsz, _ = calc_nfeats(gold_filtsz, None, None, nfeats=nfeat)
    assert filtsz == gold_filtsz


def test_use_nfeats_none():
    filtsz = [random.randint(1, 10) for _ in range(random.randint(2, 6))]
    with pytest.raises(AssertionError):
        calc_nfeats(filtsz)


def test_use_nfeats_list():
    filtsz = [random.randint(1, 10) for _ in range(random.randint(2, 6))]
    nfeats = [random.randint(1, 10) for _ in range(len(filtsz))]
    with pytest.raises(AssertionError):
        _, nfeat = calc_nfeats(filtsz, None, None, nfeats=nfeats)


def test_extract_tuple():
    filt_feat = [(random.randint(1, 10), random.randint(10, 20)) for _ in range(random.randint(2, 6))]
    gold_filtsz = tuple(filter_and_size[0] for filter_and_size in filt_feat)
    gold_nfeats = tuple(filter_and_size[1] for filter_and_size in filt_feat)
    filtsz, nfeats = calc_nfeats(filt_feat)
    assert filtsz == gold_filtsz
    assert nfeats == gold_nfeats


def test_feat_factor_manual():
    gold_filtsz = [1, 2, 3, 4, 5]
    feat_factor = 10
    gold_nfeats = [10, 20, 30, 40, 50]
    filtsz, nfeats = calc_nfeats(gold_filtsz, feat_factor, float("Infinity"))
    assert filtsz == gold_filtsz
    assert nfeats == gold_nfeats


def test_feat_factor_capped():
    gold_filtsz = [1, 2, 3, 4, 5]
    feat_factor = 10
    gold_nfeats = [10, 20, 30, 30, 30]
    filtsz, nfeats = calc_nfeats(gold_filtsz, feat_factor, 30)
    assert filtsz == gold_filtsz
    assert nfeats == gold_nfeats


def test_feat_factor():
    gold_filtsz = [random.randint(1, 10) for _ in range(random.randint(2, 6))]
    feat_factor = random.randint(10, 25)
    max_feat = random.randint(30, 40)
    gold_nfeats = np.minimum(np.array(gold_filtsz) * feat_factor, max_feat)
    filtsz, nfeats = calc_nfeats(gold_filtsz, feat_factor, max_feat)
    np.testing.assert_equal(filtsz, gold_filtsz)
    np.testing.assert_equal(nfeats, gold_nfeats)


def test_feat_factor_max_none():
    filtsz = [random.randint(1, 10) for _ in range(random.randint(2, 6))]
    feat_factor = 10
    with pytest.raises(AssertionError):
        calc_nfeats(filtsz, nfeat_factor=feat_factor, max_feat=None)

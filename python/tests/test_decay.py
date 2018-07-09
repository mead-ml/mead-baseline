import pytest
import numpy as np
from mock import patch
import baseline
from baseline.train import (
    cyclic_lr,
    zaremba_decay,
    cosine_decay,
    exponential_decay,
    staircase_decay,
    piecewise_decay,
)

@pytest.fixture
def piecewise():
    min_ = np.random.randint(1, 5)
    max_ = np.random.randint(min_ + 2, min_ + 7)
    bounds = [min_, max_]
    vals = np.random.uniform(size=len(bounds) + 1)
    return bounds, vals

def test_zaremba_values():
    eta = np.random.rand()
    decay_rate = np.random.rand()
    bounds_count = np.random.randint(5, 10)
    bounds = list(range(bounds_count))
    gold_vals = [eta / (decay_rate ** i) for i in range(len(bounds) + 1)]
    with patch('baseline.train.piecewise_decay') as pw_mock:
        zd = zaremba_decay(eta, bounds, decay_rate)
        assert pw_mock.call_args[0][0] == bounds
        assert pw_mock.call_args[0][1] == gold_vals

def test_zaremba_call_to_piecewise():
    with patch('baseline.train.piecewise_decay') as pw_mock:
        zd = zaremba_decay(1.0, [1, 2, 3], 0.5)
        pw_mock.assert_called()

def test_zaremba_with_nones():
    eta = np.random.rand()
    zd = zaremba_decay(eta)
    for step in np.random.randint(0, 1000000, size=100):
        assert zd(step) == eta

def test_piecewise_start(piecewise):
    b, v = piecewise
    p = piecewise_decay(b, v)
    lr = p(0)
    assert lr == v[0]

def test_piecewise_mid(piecewise):
    b, v = piecewise
    p = piecewise_decay(b, v)
    step = np.random.randint(np.min(b) + 1, np.max(b))
    lr = p(step)
    assert lr == v[1]

def test_piecewise_lsat(piecewise):
    b, v = piecewise
    p = piecewise_decay(b, v)
    step = np.random.randint(np.max(b) + 3, np.max(b) + 100)
    lr = p(step)
    assert lr == v[-1]

def test_staircase_decay_flat():
    steps = np.random.randint(900, 1001)
    sd = staircase_decay(np.random.rand(), steps, np.random.rand())
    stair_one_one = sd(np.random.randint(steps - 100, steps))
    stair_one_two = sd(np.random.randint(steps - 100, steps))
    stair_two = sd(np.random.randint(steps + 1, steps + 10))
    assert stair_one_one == stair_one_two
    assert stair_one_two != stair_two

def test_staircase_value():
    sd = staircase_decay(1.0, 1000, 0.9)
    gold = 1.0
    test = sd(100)
    np.testing.assert_allclose(test, gold)
    gold = 0.9
    test = sd(1001)
    np.testing.assert_allclose(test, gold)

def test_exp_values():
    sd = exponential_decay(1.0, 1000, 0.9)
    gold = 0.9895192582062144
    test = sd(100)
    np.testing.assert_allclose(test, gold)
    gold = 0.8999051805311098
    test = sd(1001)
    np.testing.assert_allclose(test, gold)

def test_cyclic_lr():
    bounds = 1000
    min_eta = 1e-5
    max_eta = 1e-2
    clr = cyclic_lr(min_eta, max_eta, bounds)
    start = clr(0)
    up = clr(bounds / 2.)
    mid = clr(bounds)
    down = clr(bounds + (bounds / 2.))
    end = clr(2 * bounds)
    late = clr(3 * bounds)
    assert start == min_eta
    assert up > start
    assert up < mid
    assert mid == max_eta
    assert down < mid
    assert down > end
    assert end == min_eta
    assert late == max_eta

def test_cosine_lr():
    cd = cosine_decay(0.1, 1000)
    iters = [0, 100, 900, 1000, 1001]
    golds = [0.1, 0.09755283, 0.002447176, 0.0, 0.0]
    for i, gold in zip(iters, golds):
        np.testing.assert_allclose(cd(i), gold, rtol=1e-6)

import six
import pytest
import numpy as np
from mock import patch, MagicMock
from eight_mile.optz import (
    create_lr_scheduler,
    CosineDecayScheduler,
    CyclicLRScheduler,
    ExponentialDecayScheduler,
    WarmupLinearScheduler,
    ConstantScheduler,
    PiecewiseDecayScheduler,
    ZarembaDecayScheduler,
    InverseTimeDecayScheduler,
    CompositeLRScheduler,
)


@pytest.fixture
def piecewise():
    min_ = np.random.randint(1, 5)
    max_ = np.random.randint(min_ + 2, min_ + 7)
    bounds = [min_, max_]
    vals = np.random.uniform(size=len(bounds) + 1)
    return bounds, vals


def test_zaremba_with_nones():
    eta = np.random.rand()
    zd = ZarembaDecayScheduler(lr=eta)
    for step in np.random.randint(0, 1000000, size=100):
        assert zd(step) == eta


def test_piecewise_start(piecewise):
    b, v = piecewise
    p = PiecewiseDecayScheduler(b, v)
    lr = p(0)
    assert lr == v[0]


def test_piecewise_mid(piecewise):
    b, v = piecewise
    p = PiecewiseDecayScheduler(b, v)
    step = np.random.randint(np.min(b) + 1, np.max(b))
    lr = p(step)
    assert lr == v[1]


def test_piecewise_lsat(piecewise):
    b, v = piecewise
    p = PiecewiseDecayScheduler(b, v)
    step = np.random.randint(np.max(b) + 3, np.max(b) + 100)
    lr = p(step)
    assert lr == v[-1]


def test_staircase_decay_flat():
    steps = np.random.randint(900, 1001)
    sd = ExponentialDecayScheduler(steps, np.random.rand(), lr=np.random.rand(), staircase=True)
    stair_one_one = sd(np.random.randint(steps - 100, steps))
    stair_one_two = sd(np.random.randint(steps - 100, steps))
    stair_two = sd(np.random.randint(steps + 1, steps + 10))
    assert stair_one_one == stair_one_two
    assert stair_one_two != stair_two


def test_staircase_value():
    sd = ExponentialDecayScheduler(1000, 0.9, lr=1.0, staircase=True)
    gold = 1.0
    test = sd(100)
    np.testing.assert_allclose(test, gold)
    gold = 0.9
    test = sd(1001)
    np.testing.assert_allclose(test, gold)


def test_exp_values():
    sd = ExponentialDecayScheduler(1000, 0.9, lr=1.0)
    gold = 0.9895192582062144
    test = sd(100)
    np.testing.assert_allclose(test, gold)
    gold = 0.8999051805311098
    test = sd(1001)
    np.testing.assert_allclose(test, gold)


def test_warmup_peaks():
    steps = np.random.randint(100, 1000)
    lr = np.random.rand()
    wls = WarmupLinearScheduler(steps, lr=lr)
    peak = wls(steps)
    assert peak == lr
    past = wls(steps + np.random.randint(100, 10000))
    assert past == lr


def test_warmup_increases():
    steps = np.random.randint(100, 1000)
    lr = np.random.rand()
    wls = WarmupLinearScheduler(steps, lr=lr)
    lrs = [wls(s) for s in range(steps)]
    last = -1
    for lr in lrs:
        assert lr > last
        last = lr


def test_cyclic_lr():
    bounds = 1000
    min_eta = 1e-5
    max_eta = 1e-2
    clr = CyclicLRScheduler(max_eta, bounds, lr=min_eta)
    start = clr(0)
    up = clr(bounds / 2.0)
    mid = clr(bounds)
    down = clr(bounds + (bounds / 2.0))
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
    cd = CosineDecayScheduler(1000, lr=0.1)
    iters = [0, 100, 900, 1000, 1001]
    golds = [0.1, 0.09755283, 0.002447176, 0.0, 0.0]
    for i, gold in zip(iters, golds):
        np.testing.assert_allclose(cd(i), gold, rtol=1e-6)


def test_constant_lr():
    lr = np.random.rand()
    lrs = ConstantScheduler(lr=lr)
    for x in np.random.randint(0, 10000000, size=np.random.randint(100, 1000)):
        assert lrs(x) == lr


def test_inverse_time_values():
    eta = 1.0
    steps = np.random.randint(1, 100)
    ti = InverseTimeDecayScheduler(steps, 1.0, lr=eta)
    for i in range(1, 5):
        lr = ti(i * steps)
        assert lr == eta / (i + 1)


def test_inverse_time_is_flat():
    steps = np.random.randint(2, 100)
    ti = InverseTimeDecayScheduler(steps, np.random.rand(), staircase=True, lr=np.random.rand())
    before = steps - np.random.randint(1, steps)
    after = steps + np.random.randint(1, steps)
    after2 = steps + np.random.randint(1, steps)
    lr_before = ti(before)
    lr_after = ti(after)
    lr_after2 = ti(after2)
    assert lr_before != lr_after
    assert lr_after == lr_after2


def test_composite_calls_warm():
    warmup_steps = np.random.randint(50, 101)
    warm = MagicMock()
    warm.warmup_steps = warmup_steps
    rest = MagicMock()
    lr = CompositeLRScheduler(warm=warm, rest=rest)
    step = np.random.randint(0, warmup_steps)
    _ = lr(step)
    warm.assert_called_once_with(step)
    rest.assert_not_called()


def test_composite_calls_rest():
    warmup_steps = np.random.randint(50, 101)
    warm = MagicMock()
    warm.warmup_steps = warmup_steps
    rest = MagicMock()
    lr = CompositeLRScheduler(warm=warm, rest=rest)
    step = np.random.randint(warmup_steps + 1, six.MAXSIZE)
    _ = lr(step)
    warm.assert_not_called()
    rest.assert_called_once_with(step - warmup_steps)


def test_composite_error():
    pytest.importorskip("torch")
    from eight_mile.pytorch.optz import CompositeLRSchedulerPyTorch

    with pytest.raises(AssertionError):
        _ = create_lr_scheduler(**{"lr_scheduler_type": ["exponential", "zaremba"]})

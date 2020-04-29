import os
import json
import pytest
import numpy as np
from eight_mile.utils import get_version

tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.skipif(get_version(tf) < 2, reason="TF1.X")
from eight_mile.optz import (
    create_lr_scheduler,
    ConstantScheduler,
    WarmupLinearScheduler,
    CyclicLRScheduler,
    PiecewiseDecayScheduler,
    ZarembaDecayScheduler,
    CosineDecayScheduler,
    InverseTimeDecayScheduler,
    ExponentialDecayScheduler,
)
import numpy as np


@pytest.fixture(scope="module")
def set_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    yield
    del os.environ["CUDA_VISIBLE_DEVICES"]


INIT_LR = 1.2
NUM_STEPS = 1000
CYCLIC_LR_CONFIG = {"lr_scheduler_type": "clr", "decay_steps": 10, "lr": INIT_LR, "max_lr": INIT_LR * 3}

INVTIME_LR_CONFIG = {"lr_scheduler_type": "invtime", "decay_rate": 0.05, "decay_steps": 1, "lr": INIT_LR}


EXP_LR_CONFIG = {"lr_scheduler_type": "exponential", "decay_rate": 0.5, "decay_steps": 100, "lr": INIT_LR}

LINEAR_WARMUP_LR_CONFIG = {"lr_scheduler_type": "warmup_linear", "warmup_steps": 20, "lr": INIT_LR}

COMPOSITE_LR_CONFIG = LINEAR_WARMUP_LR_CONFIG.copy()
COMPOSITE_LR_CONFIG.update(EXP_LR_CONFIG)
COMPOSITE_LR_CONFIG["lr_scheduler_type"] = ["warmup_linear", "exponential"]

BOUNDS = [1, 2, 4, 10, 50]
ZAREMBA_DECAY_RATE = 1.2
ZAREMBA_DECAY_VALUES = [INIT_LR / (ZAREMBA_DECAY_RATE ** i) for i in range(len(BOUNDS) + 1)]
PW_ZAREMBA_LR_CONFIG = {
    "lr_scheduler_type": "piecewise",
    "boundaries": BOUNDS,
    "values": ZAREMBA_DECAY_VALUES,
    "lr": INIT_LR,
}

ZAREMBA_LR_CONFIG = {
    "lr_scheduler_type": "zaremba",
    "boundaries": BOUNDS,
    "decay_rate": ZAREMBA_DECAY_RATE,
    "lr": INIT_LR,
}

SGDR_LR_CONFIG = {"lr_scheduler_type": "sgdr", "first_decay_steps": 100}


def test_zaremba():
    from eight_mile.tf import optz

    lr_sched = create_lr_scheduler(**ZAREMBA_LR_CONFIG)
    bl_zaremba = ZarembaDecayScheduler(**ZAREMBA_LR_CONFIG)

    lrs = []
    lrs_bl = []
    expect_lrs = []
    current_lr = INIT_LR
    for step in range(NUM_STEPS):
        lr = lr_sched(step)
        lr_bl = bl_zaremba(step)
        lrs += [lr]
        lrs_bl += [lr_bl]
        if step in BOUNDS:
            b = BOUNDS.index(step)
            current_lr = ZAREMBA_DECAY_VALUES[b]
        expect_lrs += [current_lr]
    np.allclose(expect_lrs, lrs)
    np.allclose(expect_lrs, lrs_bl)


def test_piecewise():
    from eight_mile.tf import optz

    lr_sched = create_lr_scheduler(**PW_ZAREMBA_LR_CONFIG)
    bl_piecewise = PiecewiseDecayScheduler(**PW_ZAREMBA_LR_CONFIG)

    lrs = []
    lrs_bl = []
    expect_lrs = []
    current_lr = INIT_LR
    for step in range(NUM_STEPS):
        lr = lr_sched(step)
        lrs += [lr]
        lr_bl = bl_piecewise(step)
        lrs_bl += [lr_bl]
        if step in BOUNDS:
            b = BOUNDS.index(step)
            current_lr = ZAREMBA_DECAY_VALUES[b]
        expect_lrs += [current_lr]
    np.allclose(expect_lrs, lrs)
    np.allclose(expect_lrs, lrs_bl)


def test_invtime():
    from eight_mile.tf import optz

    lr_sched = create_lr_scheduler(**INVTIME_LR_CONFIG)
    bl_invtime = InverseTimeDecayScheduler(**INVTIME_LR_CONFIG)
    decay_rate = INVTIME_LR_CONFIG["decay_rate"]

    lrs = []
    lrs_bl = []
    for step in range(NUM_STEPS):
        lr = lr_sched(step)
        lrs += [lr]
        lr_bl = bl_invtime(step)
        lrs_bl += [lr_bl]
    inv_times = [INIT_LR / (1.0 + decay_rate * t) for t in range(NUM_STEPS)]
    assert np.allclose(inv_times, lrs)
    assert np.allclose(inv_times, lrs_bl)


def test_exp():
    from eight_mile.tf import optz

    lr_sched = create_lr_scheduler(**EXP_LR_CONFIG)
    bl_exp = ExponentialDecayScheduler(**EXP_LR_CONFIG)
    decay_rate = EXP_LR_CONFIG["decay_rate"]

    lrs = []
    lrs_bl = []
    for step in range(NUM_STEPS):
        lr = lr_sched(step)
        lrs += [lr]
        lr_bl = bl_exp(step)
        lrs_bl += [lr_bl]
    inv_times = [(INIT_LR * decay_rate ** (t / 100.0)) for t in range(NUM_STEPS)]
    assert np.allclose(inv_times, lrs)
    assert np.allclose(inv_times, lrs_bl)


def test_linear_warmup():
    from eight_mile.tf import optz

    lr_sched = create_lr_scheduler(**LINEAR_WARMUP_LR_CONFIG)
    warmup_steps = LINEAR_WARMUP_LR_CONFIG["warmup_steps"]

    lrs = []
    for step in range(NUM_STEPS):
        lr = lr_sched(step)
        lrs += [lr]

    expected_lrs = [INIT_LR * min(1.0, step / warmup_steps) for step in range(NUM_STEPS)]
    assert np.allclose(expected_lrs, lrs)


def test_composite_warmup():
    from eight_mile.tf import optz

    warmup_steps = COMPOSITE_LR_CONFIG["warmup_steps"]
    decay_rate = EXP_LR_CONFIG["decay_rate"]
    lr_sched = create_lr_scheduler(**COMPOSITE_LR_CONFIG)
    lrs = [lr_sched(step) for step in range(NUM_STEPS)]

    warmup_expected = [INIT_LR * min(1.0, step / warmup_steps) for step in range(NUM_STEPS)]
    exp_expected = [(INIT_LR * decay_rate ** (t / 100.0)) for t in range(NUM_STEPS)]

    for step in range(NUM_STEPS):
        if step < warmup_steps:
            assert np.allclose(lrs[step], warmup_expected[step])
        else:
            assert np.allclose(lrs[step], exp_expected[step - warmup_steps])


def test_constant():
    from eight_mile.tf import optz

    lr_sched = create_lr_scheduler(lr=INIT_LR, lr_scheduler_type="default")
    bl_const = ConstantScheduler(lr=INIT_LR)

    for step in range(NUM_STEPS):
        lr = lr_sched(step)
        assert np.isclose(INIT_LR, lr)
        assert np.isclose(INIT_LR, bl_const(step))


def test_cyclic():
    from eight_mile.tf import optz

    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()

    lr_sched = create_lr_scheduler(**CYCLIC_LR_CONFIG)
    bl_const = CyclicLRScheduler(**CYCLIC_LR_CONFIG)

    for step in range(NUM_STEPS):
        lr = lr_sched(step)
        lr_bl = bl_const(step)
        assert np.isclose(lr, lr_bl)

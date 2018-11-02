import os
import json
import pytest
import numpy as np
tf = pytest.importorskip('tensorflow')
from baseline.tf.optz import *
from baseline.train import (ConstantScheduler,
                            WarmupLinearScheduler,
                            CyclicLRScheduler,
                            PiecewiseDecayScheduler,
                            ZarembaDecayScheduler,
                            CosineDecayScheduler,
                            InverseTimeDecayScheduler,
                            ExponentialDecayScheduler)
import numpy as np


@pytest.fixture(scope="module")
def set_cpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    yield
    del os.environ['CUDA_VISIBLE_DEVICES']


INIT_LR = 1.2
NUM_STEPS = 1000
CYCLIC_LR_CONFIG = {
    "lr_scheduler_type": "clr",
    "decay_steps": 10,
    "lr": INIT_LR,
    "max_lr": INIT_LR*3
}

INVTIME_LR_CONFIG = {
    "lr_scheduler_type": "invtime",
    "decay_rate": 0.05,
    "decay_steps": 1,
    "lr": INIT_LR
}


EXP_LR_CONFIG = {
    "lr_scheduler_type": "exponential",
    "decay_rate": 0.5,
    "decay_steps": 100,
    "lr": INIT_LR
}

LINEAR_WARMUP_LR_CONFIG = {
    "lr_scheduler_type": "warmup_linear",
    "warmup_steps": 20,
    "lr": INIT_LR
}

BOUNDS = [1, 2, 4, 10, 50]
ZAREMBA_DECAY_RATE = 1.2
ZAREMBA_DECAY_VALUES = [INIT_LR/(ZAREMBA_DECAY_RATE**i) for i in range(len(BOUNDS)+1)]
PW_ZAREMBA_LR_CONFIG = {
    "lr_scheduler_type": "piecewise",
    "bounds": BOUNDS,
    "values": ZAREMBA_DECAY_VALUES,
    "lr": INIT_LR
}

ZAREMBA_LR_CONFIG = {
    "lr_scheduler_type": "zaremba",
    "bounds": BOUNDS,
    "decay_rate": ZAREMBA_DECAY_RATE,
    "lr": INIT_LR
}

SGDR_LR_CONFIG = {
    "lr_scheduler_type": "sgdr",
    "first_decay_steps": 100
}

def test_zaremba():
    tf.reset_default_graph()
    sess = tf.Session()

    lr_sched = create_lr_scheduler(**ZAREMBA_LR_CONFIG)
    bl_zaremba = ZarembaDecayScheduler(**ZAREMBA_LR_CONFIG)
    lr_var = tf.placeholder(tf.float32, shape=(), name='lr')
    step_var = tf.placeholder(tf.int32, shape=(), name='step')

    gph = lr_sched(lr_var, step_var)
    sess.run(tf.global_variables_initializer())

    lrs = []
    lrs_bl = []
    expect_lrs = []
    current_lr = INIT_LR
    for step in range(NUM_STEPS):
        lr = sess.run(gph, feed_dict={lr_var: INIT_LR, step_var: step})
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
    tf.reset_default_graph()
    sess = tf.Session()

    lr_sched = create_lr_scheduler(**PW_ZAREMBA_LR_CONFIG)
    bl_piecewise = PiecewiseDecayScheduler(**PW_ZAREMBA_LR_CONFIG)
    lr_var = tf.placeholder(tf.float32, shape=(), name='lr')
    step_var = tf.placeholder(tf.int32, shape=(), name='step')

    gph = lr_sched(lr_var, step_var)
    sess.run(tf.global_variables_initializer())

    lrs = []
    lrs_bl = []
    expect_lrs = []
    current_lr = INIT_LR
    for step in range(NUM_STEPS):
        lr = sess.run(gph, feed_dict={lr_var: INIT_LR, step_var: step})
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
    tf.reset_default_graph()
    sess = tf.Session()

    lr_sched = create_lr_scheduler(**INVTIME_LR_CONFIG)
    bl_invtime = InverseTimeDecayScheduler(**INVTIME_LR_CONFIG)
    decay_rate = INVTIME_LR_CONFIG['decay_rate']

    lr_var = tf.placeholder(tf.float32, shape=(), name='lr')
    step_var = tf.placeholder(tf.int32, shape=(), name='step')

    gph = lr_sched(lr_var, step_var)
    sess.run(tf.global_variables_initializer())

    lrs = []
    lrs_bl = []
    for step in range(NUM_STEPS):
        lr = sess.run(gph, feed_dict={lr_var: INIT_LR, step_var: step})
        lrs += [lr]
        lr_bl = bl_invtime(step)
        lrs_bl += [lr_bl]
    inv_times = [INIT_LR / (1.0 + decay_rate * t) for t in range(NUM_STEPS)]
    print(lrs_bl[:5])
    print(lrs[:5])
    assert np.allclose(inv_times, lrs)
    assert np.allclose(inv_times, lrs_bl)


def test_exp():
    tf.reset_default_graph()
    sess = tf.Session()

    lr_sched = create_lr_scheduler(**EXP_LR_CONFIG)
    bl_exp = ExponentialDecayScheduler(**EXP_LR_CONFIG)
    decay_rate = EXP_LR_CONFIG['decay_rate']

    lr_var = tf.placeholder(tf.float32, shape=(), name='lr')
    step_var = tf.placeholder(tf.int32, shape=(), name='step')

    gph = lr_sched(lr_var, step_var)
    sess.run(tf.global_variables_initializer())

    lrs = []
    lrs_bl = []
    for step in range(NUM_STEPS):
        lr = sess.run(gph, feed_dict={lr_var: INIT_LR, step_var: step})
        lrs += [lr]
        lr_bl = bl_exp(step)
        lrs_bl += [lr_bl]
    inv_times = [(INIT_LR * decay_rate ** (t/100.)) for t in range(NUM_STEPS)]
    print(lrs_bl[:5])
    print(lrs[:5])
    assert np.allclose(inv_times, lrs)
    assert np.allclose(inv_times, lrs_bl)

def test_linear_warmup():
    tf.reset_default_graph()
    sess = tf.Session()

    lr_sched = create_lr_scheduler(**LINEAR_WARMUP_LR_CONFIG)
    warmup_steps = LINEAR_WARMUP_LR_CONFIG['warmup_steps']

    lr_var = tf.placeholder(tf.float32, shape=(), name='lr')
    step_var = tf.placeholder(tf.int32, shape=(), name='step')

    gph = lr_sched(lr_var, step_var)
    sess.run(tf.global_variables_initializer())

    lrs = []
    for step in range(NUM_STEPS):
        lr = sess.run(gph, feed_dict={lr_var: INIT_LR, step_var: step})
        lrs += [lr]

    expected_lrs = [INIT_LR*min(1.0, step / warmup_steps) for step in range(NUM_STEPS)]
    assert np.allclose(expected_lrs, lrs)


def test_constant():
    tf.reset_default_graph()
    sess = tf.Session()

    lr_sched = create_lr_scheduler(lr=INIT_LR, lr_scheduler_type='default')
    bl_const = ConstantScheduler(lr=INIT_LR)

    lr_var = tf.placeholder(tf.float32, shape=(), name='lr')
    step_var = tf.placeholder(tf.int32, shape=(), name='step')

    gph = lr_sched(lr_var, step_var)
    sess.run(tf.global_variables_initializer())

    for step in range(NUM_STEPS):
        lr = sess.run(gph, feed_dict={lr_var: INIT_LR, step_var: step})
        assert np.isclose(INIT_LR, lr)
        assert np.isclose(INIT_LR, bl_const(step))

def test_cyclic():
    tf.reset_default_graph()
    sess = tf.Session()

    lr_sched = create_lr_scheduler(**CYCLIC_LR_CONFIG)
    bl_const = CyclicLRScheduler(**CYCLIC_LR_CONFIG)

    lr_var = tf.placeholder(tf.float32, shape=(), name='lr')
    step_var = tf.placeholder(tf.int32, shape=(), name='step')

    gph = lr_sched(lr_var, step_var)
    sess.run(tf.global_variables_initializer())

    for step in range(NUM_STEPS):
        lr = sess.run(gph, feed_dict={lr_var: INIT_LR, step_var: step})
        lr_bl = bl_const(step)
        assert np.isclose(lr, lr_bl)


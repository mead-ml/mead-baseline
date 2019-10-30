import os
import pytest
import numpy as np
tf = pytest.importorskip('tensorflow')
from eight_mile.utils import get_version
pytestmark = pytest.mark.skipif(get_version(tf) >= 2, reason='tf2.0')

from baseline.tf.tfy import _add_ema


@pytest.fixture(scope="module")
def set_cpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    yield
    del os.environ['CUDA_VISIBLE_DEVICES']


class Model:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.graph.as_default():
            self.w = tf.Variable(100, dtype=tf.float32)
            self.w2 = tf.Variable(12, dtype=tf.float32)
            self.w3 = tf.Variable(64, dtype=tf.float32, trainable=False)
            self.train_vars = [self.w, self.w2]
            self.x = tf.placeholder(tf.float32, [None])
            self.y = tf.placeholder(tf.float32, [None])
            self.y_ = tf.multiply(self.x, self.w)

            self.loss = tf.reduce_sum(tf.square(tf.subtract(self.y, self.y_)))
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)


@pytest.fixture
def model():
    tf.reset_default_graph()
    m = Model()
    with m.sess.graph.as_default():
       ema_op, load, restore = _add_ema(m, 0.99)
       with tf.control_dependencies([ema_op]):
            train_op = m.opt.minimize(m.loss)
    yield m, train_op, load, restore
    m.sess.close()


def test_ema_is_graph(model):
    m, _, _, _ = model
    for var in m.train_vars:
        m.sess.graph.get_tensor_by_name("{}/ExponentialMovingAverage:0".format(var.name[:-2]))
    with pytest.raises(KeyError):
        m.sess.graph.get_tensor_by_name("{}/ExponentialMovingAverage:0".format(m.w3.name[:-2]))
    assert True


def test_trainable_in_backup(model):
    m, _, _, _ = model
    for var in m.train_vars:
        m.sess.graph.get_tensor_by_name("BackupVariables/{}".format(var.name))
    assert True


def test_restore(model):
    m, _, _, restore = model
    m.sess.run(tf.global_variables_initializer())
    old = m.sess.run(m.w)
    m.sess.run(tf.assign(m.w, tf.constant(3.0)))
    over = m.sess.run(m.w)
    m.sess.run(restore)
    new = m.sess.run(m.w)
    assert old != over
    assert over != new
    assert old == new


def test_load_saves_before_write(model):
    m, _, load, _ = model
    m.sess.run(tf.global_variables_initializer())
    old_backup = m.sess.run(m.sess.graph.get_tensor_by_name("BackupVariables/Variable:0"))
    m.sess.run(tf.assign(m.w, tf.constant(3.0)))
    gold = m.sess.run(m.w)
    m.sess.run(load)
    new_backup = m.sess.run(m.sess.graph.get_tensor_by_name("BackupVariables/Variable:0"))
    assert old_backup != new_backup
    assert new_backup == gold


# def test_load_twice_does_not_overwirte(model):
#     m, _, load, _ = model
#     m.sess.run(tf.global_variables_initializer())
#     m.sess.run(tf.assign(m.w, tf.constant(3.0)))
#     gold = m.sess.run(m.w)
#     m.sess.run(load)
#     wrong = m.sess.run(m.w)
#     m.sess.run(load)
#     backup = m.sess.run(m.sess.graph.get_tensor_by_name("BackupVariables/Variable:0"))
#     assert backup != wrong
#     assert backup == gold


def test_ema_update_on_step(model):
    m, train_op, load, _ = model
    m.sess.run(tf.global_variables_initializer())
    m.sess.run([train_op], {m.x: [1], m.y: [0]})
    m.sess.run([train_op], {m.x: [1], m.y: [0]})
    w = m.sess.run(m.w)
    m.sess.run(load)
    new_w = m.sess.run(m.w)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(w, new_w)

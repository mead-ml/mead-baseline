import os
import random
import pytest
from mock import patch, MagicMock
import numpy as np
tf = pytest.importorskip('tensorflow')
from baseline.tf.tfy import _add_ema

os.environ['CUDA_VISIBLE_DEVICES'] = ''

class Model:
    def __init__(self):
        tf.reset_default_graph()
        self.w = tf.Variable(0, dtype=tf.float32)
        self.w2 = tf.Variable(12, dtype=tf.float32)
        self.w3 = tf.Variable(65, dtype=tf.float32, trainable=False)
        self.train_vars = [self.w, self.w2]
        self.x = tf.placeholder(tf.float32, [None])
        self.y = tf.placeholder(tf.float32, [None])
        self.y_ = tf.multiply(self.x, self.w)

        self.loss = tf.reduce_sum(tf.square(tf.subtract(self.y, self.y_)))
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        self.sess = tf.Session()

@pytest.fixture
def model():
    m = Model()
    m.ema, m.ema_op, m.eval_saver = _add_ema(m, 0.99)
    with tf.control_dependencies([m.ema_op]):
        train_op = m.opt.minimize(m.loss)
    return m, train_op

@pytest.fixture
def save_file():
    loc = os.path.dirname(os.path.realpath(__file__))
    file_name = os.path.join(loc, 'test_data', 'ema')
    yield file_name
    dir_name, file_prefix = os.path.split(file_name)
    for file_name in os.listdir(dir_name):
        if file_name.startswith(file_prefix):
            try:
                os.remove(os.path.join(dir_name, file_name))
            except OSError:
                pass
    try:
        os.remove(os.path.join(dir_name, "checkpoint"))
    except OSError:
        pass

def test_ema_in_graph(model):
    model, _ = model
    for var in model.train_vars:
        model.sess.graph.get_tensor_by_name("{}:0".format(model.ema.average_name(var)))
    with pytest.raises(KeyError):
        model.sess.graph.get_tensor_by_name("{}:0".format(model.ema.average_name(model.w3)))
    assert True

def test_ema_updated(model):
    # Set EMA
    model, train_op = model
    model.sess.run(tf.global_variables_initializer())

    # Get initial values of EMA
    prev_w = model.sess.run(
        model.sess.graph.get_tensor_by_name(
            "{}:0".format(model.ema.average_name(model.w))
        )
    )
    prev_w2 = model.sess.run(
        model.sess.graph.get_tensor_by_name(
            "{}:0".format(model.ema.average_name(model.w2))
        )
    )

    # Run training op (twice so ema has something moving)
    model.sess.run([train_op], {model.x: [1], model.y: [10]})
    model.sess.run([train_op], {model.x: [1], model.y: [10]})

    # Get new values of EMA
    new_w = model.sess.run(model.sess.graph.get_tensor_by_name("{}:0".format(model.ema.average_name(model.w))))
    new_w2 = model.sess.run(model.sess.graph.get_tensor_by_name("{}:0".format(model.ema.average_name(model.w2))))
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(prev_w, new_w)
    np.testing.assert_allclose(prev_w2, new_w2)

def test_ema_in_reload(model, save_file):
    model, _ = model
    model.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    saver.save(model.sess, save_file)
    saver.restore(model.sess, save_file)

    for var in model.train_vars:
        model.sess.graph.get_tensor_by_name("{}:0".format(model.ema.average_name(var)))


def test_ema_updates_after_reload(model, save_file):
    model, train_op = model
    model.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    model.sess.run([train_op], {model.x: [1], model.y: [10]})
    model.sess.run([train_op], {model.x: [1], model.y: [10]})

    saver.save(model.sess, save_file)
    saver.restore(model.sess, save_file)

    # Get initial values of EMA
    prev_w = model.sess.run(
        model.sess.graph.get_tensor_by_name(
            "{}:0".format(model.ema.average_name(model.w))
        )
    )
    prev_w2 = model.sess.run(
        model.sess.graph.get_tensor_by_name(
            "{}:0".format(model.ema.average_name(model.w2))
        )
    )

    # Run training op (twice so ema has something moving)
    model.sess.run([train_op], {model.x: [1], model.y: [10]})
    model.sess.run([train_op], {model.x: [1], model.y: [10]})

    # Get new values of EMA
    new_w = model.sess.run(
        model.sess.graph.get_tensor_by_name(
            "{}:0".format(model.ema.average_name(model.w))
        )
    )
    new_w2 = model.sess.run(
        model.sess.graph.get_tensor_by_name(
            "{}:0".format(model.ema.average_name(model.w2))
        )
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(prev_w, new_w)
    np.testing.assert_allclose(prev_w2, new_w2)

def test_ema_values_are_used_when_restore_with_eval_saver(model, save_file):
    model, train_op = model
    model.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    model.sess.run([train_op], {model.x: [1], model.y: [10]})
    model.sess.run([train_op], {model.x: [1], model.y: [10]})

    old_w = model.sess.run(model.w)
    old_w_ema = model.sess.run(
        model.sess.graph.get_tensor_by_name(
            "{}:0".format(model.ema.average_name(model.w))
        )
    )

    saver.save(model.sess, save_file)
    model.eval_saver.restore(model.sess, save_file)

    new_w = model.sess.run(model.w)

    np.testing.assert_allclose(old_w_ema, new_w)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(old_w, new_w)

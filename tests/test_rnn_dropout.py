import os
import pytest

pytest.skip("This has been broken for a while, will fix soon, BL", allow_module_level=True)
import numpy as np

tf = pytest.importorskip("tensorflow")
from baseline.tf.tfy import rnn_cell_w_dropout, lstm_cell_w_dropout


@pytest.fixture(scope="module")
def set_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    yield
    del os.environ["CUDA_VISIBLE_DEVICES"]


def test_static_dropout_lstm_cell():
    with tf.device("/cpu:0"):
        sess = tf.compat.v1.Session()
        x = np.random.randn(1, 10, 50).astype(np.float32)
        with sess.graph.as_default():
            with tf.variable_scope("DropoutIsOn"):
                rnn_drop_cell = lstm_cell_w_dropout(100, 0.9999999999, training=True)
                rnn_drop, _ = tf.nn.dynamic_rnn(
                    rnn_drop_cell, x, sequence_length=np.array([10], dtype=np.int), dtype=tf.float32
                )
            with tf.variable_scope("DropoutIsOff"):
                rnn_no_drop_cell = lstm_cell_w_dropout(100, 0.9999999999, training=False)
                rnn_no_drop, _ = tf.nn.dynamic_rnn(
                    rnn_no_drop_cell, x, sequence_length=np.array([10], dtype=np.int), dtype=tf.float32
                )
        sess.run(tf.compat.v1.global_variables_initializer())
        out_ten = sess.run(rnn_drop)
        assert len(out_ten[np.nonzero(out_ten)].squeeze()) < 20
        out_ten = sess.run(rnn_no_drop)
        assert len(out_ten[np.nonzero(out_ten)].squeeze()) > 20


def test_static_dropout_rnn_cell():
    with tf.device("/cpu:0"):
        sess = tf.compat.v1.Session()
        x = np.random.randn(1, 10, 50).astype(np.float32)
        with sess.graph.as_default():
            with tf.variable_scope("DropoutIsOn"):
                rnn_drop_cell = rnn_cell_w_dropout(100, 0.9999999999, "gru", training=True)
                rnn_drop, _ = tf.nn.dynamic_rnn(
                    rnn_drop_cell, x, sequence_length=np.array([10], dtype=np.int), dtype=tf.float32
                )
            with tf.variable_scope("DropoutIsOff"):
                rnn_no_drop_cell = rnn_cell_w_dropout(100, 0.9999999999, "gru", training=False)
                rnn_no_drop, _ = tf.nn.dynamic_rnn(
                    rnn_no_drop_cell, x, sequence_length=np.array([10], dtype=np.int), dtype=tf.float32
                )
        sess.run(tf.compat.v1.global_variables_initializer())
        out_ten = sess.run(rnn_drop)
        assert len(out_ten[np.nonzero(out_ten)].squeeze()) < 20
        out_ten = sess.run(rnn_no_drop)
        assert len(out_ten[np.nonzero(out_ten)].squeeze()) > 20


def test_placeholder_dropout_lstm_cell():
    with tf.device("/cpu:0"):
        sess = tf.compat.v1.Session()
        x = np.random.randn(1, 10, 50).astype(np.float32)
        with sess.graph.as_default():
            train_flag = tf.compat.v1.placeholder_with_default(False, shape=(), name="TEST_TRAIN_FLAG")
            with tf.variable_scope("DropoutMightBeOn"):
                rnn_cell = lstm_cell_w_dropout(100, 0.9999999999, training=train_flag)
                rnn, _ = tf.nn.dynamic_rnn(rnn_cell, x, sequence_length=np.array([10], dtype=np.int), dtype=tf.float32)

        sess.run(tf.compat.v1.global_variables_initializer())
        out_ten = sess.run(rnn, {train_flag: True})
        assert len(out_ten[np.nonzero(out_ten)].squeeze()) < 20
        out_ten = sess.run(rnn)
        assert len(out_ten[np.nonzero(out_ten)].squeeze()) > 20


def test_placeholder_dropout_rnn_cell():
    with tf.device("/cpu:0"):
        sess = tf.compat.v1.Session()
        x = np.random.randn(1, 10, 50).astype(np.float32)
        with sess.graph.as_default():
            train_flag = tf.compat.v1.placeholder_with_default(False, shape=(), name="TEST_TRAIN_FLAG")
            with tf.variable_scope("DropoutMightBeOn"):
                rnn_cell = rnn_cell_w_dropout(100, 0.9999999999, "gru", training=train_flag)
                rnn, _ = tf.nn.dynamic_rnn(rnn_cell, x, sequence_length=np.array([10], dtype=np.int), dtype=tf.float32)

        sess.run(tf.compat.v1.global_variables_initializer())
        out_ten = sess.run(rnn, {train_flag: True})
        assert len(out_ten[np.nonzero(out_ten)].squeeze()) < 20
        out_ten = sess.run(rnn)
        assert len(out_ten[np.nonzero(out_ten)].squeeze()) > 20

import os
import pytest
import numpy as np

tf = pytest.importorskip("tensorflow")
from eight_mile.utils import get_version
from eight_mile.tf.layers import SeqDotProductAttention, SeqScaledDotProductAttention, subsequent_mask


@pytest.fixture(scope="module")
def set_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    yield
    del os.environ["CUDA_VISIBLE_DEVICES"]


@pytest.fixture
def qkv():
    with tf.device("/cpu:0"):
        dim = np.random.randint(5, 10, size=4)
        q = tf.random.normal(shape=dim)
        k = tf.random.normal(shape=dim)
        v = tf.random.normal(shape=dim)
    return q, k, v


def test_attn_value(qkv):
    q, k, v = qkv
    with tf.device("/cpu:0"):
        q = tf.zeros_like(q)
        dot_product_attention = SeqDotProductAttention(0.0)
        res = dot_product_attention((q, k, v, None))
        if get_version(tf) < 2:
            with tf.compat.v1.Session() as sess:
                res, gold = sess.run([res, v])
        else:
            res, gold = res.numpy(), v.numpy()
        B, H, T, _ = q.get_shape().as_list()
        for b in range(B):
            for h in range(H):
                for t in range(T):
                    np.testing.assert_allclose(res[b, h, t, :], np.mean(gold, axis=2)[b, h, :], atol=1e-5)


@pytest.mark.skipif(get_version(tf) < 2, reason="needs tf2")
def test_attn_value_seq_mask(qkv):
    q, k, v = qkv
    with tf.device("/cpu:0"):
        B, H, T, _ = q.get_shape().as_list()
        q = tf.zeros_like(q)
        lens = np.random.randint(1, T, size=B).astype(np.int32)
        tf_lens = tf.constant(lens)
        mask = tf.expand_dims(tf.expand_dims(tf.sequence_mask(tf_lens, T, dtype=tf.float32), 1), 1)
        dot_product_attention = SeqDotProductAttention(0.0)
        res = dot_product_attention((q, k, v, mask))
        res, gold = res.numpy(), v.numpy()
        for b in range(B):
            for h in range(H):
                for t in range(T):
                    np.testing.assert_allclose(
                        res[b, h, t, :], np.mean(gold[:, :, : lens[b], :], axis=2)[b, h, :], atol=1e-5
                    )


@pytest.mark.skipif(get_version(tf) < 2, reason="needs tf2")
def test_attn_value_sub_mask(qkv):
    q, k, v = qkv
    with tf.device("/cpu:0"):
        B, H, T, _ = q.get_shape().as_list()
        q = tf.zeros_like(q)
        mask = subsequent_mask(T)
        dot_product_attention = SeqDotProductAttention(0.0)
        res = dot_product_attention((q, k, v, mask))
        res, gold = res.numpy(), v.numpy()
        for b in range(B):
            for h in range(H):
                for t in range(T):
                    np.testing.assert_allclose(
                        res[b, h, t, :], np.mean(gold[:, :, : t + 1, :], axis=2)[b, h, :], atol=1e-5
                    )


def test_scaled_attn_value(qkv):
    q, k, v = qkv
    with tf.device("/cpu:0"):
        q = tf.zeros_like(q)
        scaled_dot_product_attention = SeqScaledDotProductAttention(0.0)
        res = scaled_dot_product_attention((q, k, v, None))
        if get_version(tf) < 2:
            with tf.compat.v1.Session() as sess:
                res, gold = sess.run([res, v])
        else:
            res, gold = res.numpy(), v.numpy()
        B, H, T, _ = q.get_shape().as_list()
        for b in range(B):
            for h in range(H):
                for t in range(T):
                    np.testing.assert_allclose(res[b, h, t, :], np.mean(gold, axis=2)[b, h, :], atol=1e-5)


@pytest.mark.skipif(get_version(tf) < 2, reason="needs tf2")
def test_scaled_attn_value_seq_mask(qkv):
    q, k, v = qkv
    with tf.device("/cpu:0"):
        B, H, T, _ = q.get_shape().as_list()
        q = tf.zeros_like(q)
        lens = np.random.randint(1, T, size=B).astype(np.int32)
        tf_lens = tf.constant(lens)
        mask = tf.expand_dims(tf.expand_dims(tf.sequence_mask(tf_lens, T, dtype=tf.float32), 1), 1)
        scaled_dot_product_attention = SeqScaledDotProductAttention(0.0)
        res = scaled_dot_product_attention((q, k, v, mask))
        res, gold = res.numpy(), v.numpy()
        for b in range(B):
            for h in range(H):
                for t in range(T):
                    np.testing.assert_allclose(
                        res[b, h, t, :], np.mean(gold[:, :, : lens[b], :], axis=2)[b, h, :], atol=1e-5
                    )


@pytest.mark.skipif(get_version(tf) < 2, reason="needs tf2")
def test_scaled_attn_value_sub_mask(qkv):
    q, k, v = qkv
    with tf.device("/cpu:0"):
        B, H, T, _ = q.get_shape().as_list()
        q = tf.zeros_like(q)
        mask = subsequent_mask(T)
        scaled_dot_product_attention = SeqScaledDotProductAttention(0.0)
        res = scaled_dot_product_attention((q, k, v, mask))
        res, gold = res.numpy(), v.numpy()
        for b in range(B):
            for h in range(H):
                for t in range(T):
                    np.testing.assert_allclose(
                        res[b, h, t, :], np.mean(gold[:, :, : t + 1, :], axis=2)[b, h, :], atol=1e-5
                    )

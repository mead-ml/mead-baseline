import pytest
import numpy as np
tf = pytest.importorskip('tensorflow')
from baseline.tf.transformer import dot_product_attention, subsequent_mask


@pytest.fixture(scope="module")
def set_cpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    yield
    del os.environ['CUDA_VISIBLE_DEVICES']


@pytest.fixture(scope="function")
def reset():
    tf.reset_default_graph()


@pytest.fixture
def qkv():
    dim = np.random.randint(5, 10, size=4)
    q = tf.random_normal(shape=dim)
    k = tf.random_normal(shape=dim)
    v = tf.random_normal(shape=dim)
    return q, k, v


def test_attn_value(qkv):
    q, k, v = qkv
    q = tf.zeros_like(q)
    res = dot_product_attention(q, k, v)
    with tf.Session() as sess:
        res, gold = sess.run([res, v])
    B, H, T, _ = q.get_shape().as_list()
    for b in range(B):
        for h in range(H):
            for t in range(T):
                np.testing.assert_allclose(res[b, h, t, :], np.mean(gold, axis=2)[b, h, :], atol=1e-5)


def test_attn_value_seq_mask(qkv):
    q, k, v = qkv
    B, H, T, _ = q.get_shape().as_list()
    q = tf.zeros_like(q)
    lens = np.random.randint(1, T, size=B).astype(np.int32)
    tf_lens = tf.constant(lens)
    mask = tf.expand_dims(tf.expand_dims(tf.sequence_mask(tf_lens, T, dtype=tf.float32), 1), 1)
    res = dot_product_attention(q, k, v, mask=mask)
    with tf.Session() as sess:
        res, gold = sess.run([res, v])
    for b in range(B):
        for h in range(H):
            for t in range(T):
                print(b, h, t)
                np.testing.assert_allclose(res[b, h, t, :], np.mean(gold[:, :, :lens[b], :], axis=2)[b, h, :], atol=1e-5)


def test_attn_value_sub_mask(qkv):
    q, k, v = qkv
    B, H, T, _ = q.get_shape().as_list()
    q = tf.zeros_like(q)
    mask = subsequent_mask(T)
    res = dot_product_attention(q, k, v, mask=mask)
    with tf.Session() as sess:
        res, gold = sess.run([res, v])
    for b in range(B):
        for h in range(H):
            for t in range(T):
                np.testing.assert_allclose(res[b, h, t, :], np.mean(gold[:, :, :t+1, :], axis=2)[b, h, :], atol=1e-5)

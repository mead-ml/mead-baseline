import pytest
import os
import numpy as np
tf = pytest.importorskip("tensorflow")
from eight_mile.utils import get_version
from eight_mile.tf.layers import SeqScaledDotProductRelativeAttention, SeqScaledWindowedRelativeAttention, SET_TRAIN_FLAG, masked_fill


@pytest.fixture(scope="module")
def set_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    yield
    del os.environ["CUDA_VISIBLE_DEVICES"]


def make_rpr(rpr_key_emb, rpr_value_emb, rpr_k, seq_len):
    """Create a matrix shifted by self.rpr_k and bounded between 0 and 2*self.rpr_k to provide 0-based indexing for embedding
    """
    seq = tf.range(seq_len)
    window_len = 2 * rpr_k
    edges = tf.reshape(seq, [1, -1]) - tf.reshape(seq, [-1, 1]) + rpr_k
    edges = tf.clip_by_value(edges, 0, window_len)
    return rpr_key_emb(edges), rpr_value_emb(edges)


def unfold_rpr(rpr_key_emb, rpr_value_emb, rpr_k):
    window_len = 2 * rpr_k + 1
    window = tf.range(window_len)
    return rpr_key_emb(window), rpr_value_emb(window)


@pytest.mark.skipif(get_version(tf) < 2, reason="needs tf2")
def test_windowed_ra():
    num_heads = 4
    d_model = 64
    rpr_k = 1
    batchsize = 2
    nctx = 256
    d_k = d_model // num_heads

    with tf.device("/cpu:0"):
        old = SeqScaledDotProductRelativeAttention(pdrop=0.)
        new = SeqScaledWindowedRelativeAttention(pdrop=0.)

        rpr_key_emb = tf.keras.layers.Embedding(2 * rpr_k + 1, d_k)
        rpr_value_emb = tf.keras.layers.Embedding(2 * rpr_k + 1, d_k)

        Q = tf.random.normal([batchsize, num_heads, nctx, d_k])
        K = tf.random.normal([batchsize, num_heads, nctx, d_k])
        V = tf.random.normal([batchsize, num_heads, nctx, d_k])
        lengths = tf.random.uniform([batchsize, ], 0, nctx, dtype=tf.int32)
        seq_mask = tf.sequence_mask(lengths, maxlen=nctx, dtype=tf.float32)
        in_mask = tf.expand_dims(tf.expand_dims(seq_mask, 1), 1)
        out_mask = tf.expand_dims(tf.expand_dims(seq_mask, 1), -1)

        # manually create a ra_mask to prevent attention beyond rpr_k
        ones = tf.ones([nctx, nctx])
        ra_mask = tf.linalg.band_part(ones, rpr_k, rpr_k)
        mask = in_mask * tf.expand_dims(tf.expand_dims(ra_mask, 0), 0)
        rpr_key_old, rpr_value_old = make_rpr(rpr_key_emb, rpr_value_emb, rpr_k, nctx)
        SET_TRAIN_FLAG(False)
        out_old = old((Q, K, V, rpr_key_old, rpr_value_old, mask))
        out_old = masked_fill(out_old, tf.equal(out_mask, 0), 1)
        print(out_old.shape)

        # using the windowed relative attention with the original sequence mask
        rpr_key_new, rpr_value_new = unfold_rpr(rpr_key_emb, rpr_value_emb, rpr_k)
        out_new = new((Q, K, V, rpr_key_new, rpr_value_new, in_mask))
        out_new = masked_fill(out_new, tf.equal(out_mask, 0), 1)
        print(out_new.shape)
        if get_version(tf) < 2:
            with tf.compat.v1.Session() as sess:
                out_old, out_new = sess.run([out_old, out_new])
        else:
            out_old, out_new = out_old.numpy(), out_new.numpy()

        assert np.allclose(out_old, out_new, atol=1e-6)

import pytest
import numpy as np

#try:
#    import tensorflow as tf
#except:
#    pass
tf = pytest.importorskip("tensorflow")
tf.compat.v1.enable_eager_execution()

from eight_mile.utils import Offsets
from eight_mile.tf.layers import *

C = 10
B = 50
T = 20
H = 8
L = 2

@pytest.fixture
def lengths():
    lengths = tf.cast(np.random.randint(1, T, size=(B,)), tf.int32)
    return lengths


@pytest.fixture
def logits(lengths):
    logits = np.random.rand(B, T, C)
    for i, l in enumerate(lengths):
        logits[i, l:, :] = 0
    return tf.cast(logits, tf.float32)


@pytest.fixture
def labels(lengths):
    lab = np.random.randint(1, C, size=(B, T))
    for i, l in enumerate(lengths):
        lab[i, l:] = 0
    return tf.cast(lab, tf.int32)

@pytest.fixture
def encoder_input(lengths):
    hidden = tf.cast(np.random.rand(B, T, C), tf.float32)
    return hidden, lengths

def test_gru_sequence_shapes(encoder_input):

    z = GRUEncoderSequence(C, H, L, batch_first=True)
    out = z(encoder_input)
    seq_len = tf.reduce_max(encoder_input[1]).numpy().item()
    np.testing.assert_equal(out.shape, (B, seq_len, H))

    z = BiGRUEncoderSequence(C, H, L, batch_first=True)
    out = z(encoder_input)
    np.testing.assert_equal(out.shape, (B, seq_len, H))


def test_gru_hidden_shapes(encoder_input):

    z = GRUEncoderHidden(C, H, L, batch_first=True)
    out = z(encoder_input)
    np.testing.assert_equal(out.shape, (B, H))

    z = BiGRUEncoderHidden(C, H, L, batch_first=True)
    out = z(encoder_input)
    np.testing.assert_equal(out.shape, (B, H))


def test_gru_all_shapes(encoder_input):

    seq_len = tf.reduce_max(encoder_input[1]).numpy().item()
    z = GRUEncoderAll(C, H, L, batch_first=True)
    seq, s = z(encoder_input)
    np.testing.assert_equal(seq.shape, (B, seq_len, H))
    np.testing.assert_equal(s.shape, (L, B, H))

    z = BiGRUEncoderAll(C, H, L, batch_first=True)
    seq, s = z(encoder_input)
    np.testing.assert_equal(seq.shape, (B, seq_len, H))
    np.testing.assert_equal(s.shape, (L, B, H))


def test_lstm_sequence_shapes(encoder_input):

    z = LSTMEncoderSequence(C, H, L, batch_first=True)
    out = z(encoder_input)
    seq_len = tf.reduce_max(encoder_input[1]).numpy().item()
    np.testing.assert_equal(out.shape, (B, seq_len, H))

    z = BiLSTMEncoderSequence(C, H, L, batch_first=True)
    out = z(encoder_input)
    np.testing.assert_equal(out.shape, (B, seq_len, H))


def test_lstm_hidden_shapes(encoder_input):

    z = LSTMEncoderHidden(C, H, L, batch_first=True)
    out = z(encoder_input)
    np.testing.assert_equal(out.shape, (B, H))

    z = BiLSTMEncoderHidden(C, H, L, batch_first=True)
    out = z(encoder_input)
    np.testing.assert_equal(out.shape, (B, H))



def test_lstm_all_shapes(encoder_input):

    seq_len = tf.reduce_max(encoder_input[1]).numpy().item()
    # There is an interface mismatch between v1 and v2
    z = LSTMEncoderAll(C, H, L, batch_first=True)
    seq, (s1, s2) = z(encoder_input)
    np.testing.assert_equal(seq.shape, (B, seq_len, H))
    np.testing.assert_equal(s1.shape, (L, B, H))
    np.testing.assert_equal(s2.shape, (L, B, H))

    z = BiLSTMEncoderAll(C, H, L, batch_first=True)
    seq, (s1, s2) = z(encoder_input)
    np.testing.assert_equal(seq.shape, (B, seq_len, H))
    np.testing.assert_equal(s1.shape, (L, B, H))
    np.testing.assert_equal(s2.shape, (L, B, H))

def test_conv_shapes(encoder_input):

    z = WithoutLength(ConvEncoder(C, H, 4))
    seq = z(encoder_input)
    np.testing.assert_equal(seq.shape, (B, T, H))

    z = WithoutLength(ConvEncoder(C, H, 3))
    seq = z(encoder_input)
    np.testing.assert_equal(seq.shape, (B, T, H))

    z = WithoutLength(ConvEncoderStack(C, H, 3, nlayers=3))
    seq = z(encoder_input)
    np.testing.assert_equal(seq.shape, (B, T, H))

    z = WithoutLength(ConvEncoderStack(C, H, 4, nlayers=3))
    seq = z(encoder_input)
    np.testing.assert_equal(seq.shape, (B, T, H))

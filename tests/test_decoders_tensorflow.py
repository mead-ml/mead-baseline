import os
import math
import json
import pytest
import numpy as np
from eight_mile.utils import get_version
from eight_mile.embeddings import RandomInitVecModel
from collections import namedtuple
import string
tf = pytest.importorskip('tensorflow')
pytestmark = pytest.mark.skipif(get_version(tf) < 2, reason='TF1.X')
from eight_mile.utils import Offsets


def test_rnn_decode_shapes():
    from baseline.tf.embeddings import LookupTableEmbeddingsModel
    from baseline.tf.seq2seq.decoders.v2 import RNNDecoder
    # Always pick the right path
    encoder = namedtuple("EncoderOutput", "output src_mask")
    batchsz = 2
    temporal = 7
    temporal_output = 4
    hsz = 20
    dsz = 10
    layers = 1
    # Always pick the right path
    wv = RandomInitVecModel(
        dsz, {k: 1 for k in list(string.ascii_letters)}
    )
    assert len(string.ascii_letters) + len(Offsets.VALUES) == wv.get_vsz()
    encoder.output = tf.cast(np.random.randn(batchsz, temporal, hsz), dtype=tf.float32)
    encoder.hidden = (tf.cast(np.random.randn(layers, batchsz, hsz), dtype=tf.float32),
                      tf.cast(np.random.randn(layers, batchsz, hsz), dtype=tf.float32))
    encoder.src_mask = np.zeros((batchsz, temporal), dtype=np.uint8)
    tgt_embed = LookupTableEmbeddingsModel.create(wv, 'output')
    decoder = RNNDecoder(tgt_embed, hsz=hsz, tie_weights=False)
    decode_start = np.full((batchsz, temporal_output), Offsets.GO, dtype=np.int64)
    output = decoder(encoder, decode_start)
    assert output.shape[0] == batchsz
    assert output.shape[1] == temporal_output
    assert output.shape[2] == wv.get_vsz()



def test_rnn_attn_decode_shapes():
    from baseline.tf.embeddings import LookupTableEmbeddingsModel
    from baseline.tf.seq2seq.decoders.v2 import RNNDecoderWithAttn
    # Always pick the right path
    encoder = namedtuple("EncoderOutput", "output src_mask")
    batchsz = 2
    temporal = 7
    temporal_output = 4
    hsz = 20
    dsz = 10
    layers = 1
    # Always pick the right path
    wv = RandomInitVecModel(
        dsz, {k: 1 for k in list(string.ascii_letters)}
    )
    assert len(string.ascii_letters) + len(Offsets.VALUES) == wv.get_vsz()
    encoder.output = tf.cast(np.random.randn(batchsz, temporal, hsz), dtype=tf.float32)
    encoder.hidden = (tf.cast(np.random.randn(layers, batchsz, hsz), dtype=tf.float32),
                      tf.cast(np.random.randn(layers, batchsz, hsz), dtype=tf.float32))
    encoder.src_mask = np.zeros((batchsz, temporal), dtype=np.uint8)
    tgt_embed = LookupTableEmbeddingsModel.create(wv, 'output')
    decoder = RNNDecoderWithAttn(tgt_embed, hsz=hsz, attn_type='sdpx', tie_weights=False)
    decode_start = np.full((batchsz, temporal_output), Offsets.GO, dtype=np.int64)
    output = decoder(encoder, decode_start)
    assert output.shape[0] == batchsz
    assert output.shape[1] == temporal_output
    assert output.shape[2] == wv.get_vsz()

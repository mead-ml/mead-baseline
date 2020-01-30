import os
import math
import json
import pytest
import numpy as np
from eight_mile.w2v import RandomInitVecModel
from collections import namedtuple
import string
torch = pytest.importorskip('torch')
from eight_mile.utils import Offsets


def test_rnn_decode_shapes():
    from baseline.pytorch.embeddings import LookupTableEmbeddingsModel
    from baseline.pytorch.seq2seq.decoders import RNNDecoder
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
    encoder.output = np.random.randn(batchsz, temporal, hsz)
    encoder.output = torch.from_numpy(encoder.output).float()
    encoder.hidden = (torch.from_numpy(np.random.randn(layers, batchsz, hsz)).float(),
                      torch.from_numpy(np.random.randn(layers, batchsz, hsz)).float())
    encoder.output = encoder.output
    encoder.src_mask = torch.zeros(batchsz, temporal).byte()
    tgt_embed = LookupTableEmbeddingsModel.create(wv, 'output')
    decoder = RNNDecoder(tgt_embed, hsz=hsz, tie_weights=False)
    decode_start = torch.full((batchsz, temporal_output), Offsets.GO, dtype=torch.long)
    output = decoder(encoder, decode_start)
    assert output.shape[0] == batchsz
    assert output.shape[1] == temporal_output
    assert output.shape[2] == wv.get_vsz()



def test_rnn_attn_decode_shapes():
    from baseline.pytorch.embeddings import LookupTableEmbeddingsModel
    from baseline.pytorch.seq2seq.decoders import RNNDecoderWithAttn
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
    encoder.output = np.random.randn(batchsz, temporal, hsz)
    encoder.output = torch.from_numpy(encoder.output).float()
    encoder.hidden = (torch.from_numpy(np.random.randn(layers, batchsz, hsz)).float(),
                  torch.from_numpy(np.random.randn(layers, batchsz, hsz)).float())
    encoder.output = encoder.output
    encoder.src_mask = torch.zeros(batchsz, temporal).byte()

    tgt_embed = LookupTableEmbeddingsModel.create(wv, 'output')
    decoder = RNNDecoderWithAttn(tgt_embed, hsz=hsz, tie_weights=False)
    decode_start = torch.full((batchsz, temporal_output), Offsets.GO, dtype=torch.long)
    output = decoder(encoder, decode_start)
    assert output.shape[0] == batchsz
    assert output.shape[1] == temporal_output
    assert output.shape[2] == wv.get_vsz()

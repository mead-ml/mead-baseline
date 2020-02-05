import os
import math
import json
import pytest
import numpy as np
from mock import patch, MagicMock

torch = pytest.importorskip("torch")


data_loc = os.path.realpath(os.path.dirname(__file__))
data_loc = os.path.join(data_loc, "test_data")

from baseline.pytorch.lm import TransformerLanguageModel
import baseline.pytorch.embeddings
import baseline.embeddings
import torch
from eight_mile.optz import *
from eight_mile.pytorch.optz import *
from eight_mile.pytorch.layers import TransformerEncoderStack as PytTransformerEncoderStack, Dense, subsequent_mask
from eight_mile.pytorch.serialize import save_tlm_npz, load_tlm_npz
import numpy as np


file_loc = os.path.realpath(os.path.dirname(__file__))


def _call_model(m, inputs):
    from eight_mile.pytorch.layers import sequence_mask
    m.eval()
    lengths = inputs.get('lengths')
    x = m.embeddings(inputs)
    max_seqlen = x.shape[1]
    mask = sequence_mask(lengths, max_seqlen).to(x.device).unsqueeze(1).unsqueeze(1)
    return m.transformer((x, mask))


def test_round_trip():
    test_file = os.path.join(file_loc, "test_data", "blah.npz")
    d_model = 40
    vocab_x = {'a':1, 'aardvark':100, 'beandip':42, 'cheerio':86, 'dumdum':129, 'eoyre':3}
    embeddings = {}
    vocabs = {'x': vocab_x}
    x_embedding = baseline.embeddings.load_embeddings('x',
                                                      dsz=d_model,
                                                      known_vocab=vocab_x,
                                                      embed_type='positional')
    vocabs['x'] = x_embedding['vocab']
    embeddings['x'] = x_embedding['embeddings']

    src_model = TransformerLanguageModel.create(embeddings,
                                                hsz=d_model,
                                                dropout=0.1,
                                                gpu=False,
                                                num_heads=4,
                                                layers=2,
                                                src_keys=['x'], tgt_key='x')

    save_tlm_npz(src_model, test_file)
    dst_model = TransformerLanguageModel.create(embeddings,
                                                hsz=d_model,
                                                dropout=0.1,
                                                gpu=False,
                                                num_heads=4,
                                                layers=2,
                                                src_keys=['x'], tgt_key='x')
    load_tlm_npz(dst_model, test_file)


    B = 4
    T = 7
    a_batch = torch.randint(0, 9, (B, T)).long()
    a_lengths = torch.randint(0, T, (B,)).long()
    out_pyt1 = _call_model(dst_model, {'x': a_batch, 'lengths': a_lengths}).detach().numpy()
    out_pyt2 = _call_model(dst_model, {'x': a_batch, 'lengths': a_lengths}).detach().numpy()
    print(np.allclose(out_pyt1, out_pyt2, atol=1e-6))

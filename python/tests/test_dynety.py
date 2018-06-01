import random
import numpy as np
import dynet as dy
from baseline.dy.dynety import *

SIZES = [128, 256, 300, 512]
FILTER_SIZES = [3, 4, 5]


def test_linear_params_present():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    linear = Linear(12, 6, pc)
    names = {p.name() for p in pc.parameters_list()}
    assert "/Linear/Weight" in names
    assert "/Linear/Bias" in names

def test_linear_params_rename():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    gold = "TESTING"
    linear = Linear(12, 6, pc, name=gold)
    names = {p.name() for p in pc.parameters_list()}
    assert "/{}/Weight".format(gold) in names
    assert "/{}/Bias".format(gold) in names

def test_linear_param_shapes():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    out = random.choice(SIZES)
    in_ = random.choice(SIZES)
    linear = Linear(out, in_, pc)
    params = {p.name(): p for p in pc.parameters_list()}
    assert params['/Linear/Weight'].shape() == (out, in_)
    assert params['/Linear/Bias'].shape() == (out,)

def test_linear_forward_shape():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    out = random.choice(SIZES)
    in_ = random.choice(SIZES)
    linear = Linear(out, in_, pc)
    input_ = dy.inputVector(np.random.randn(in_))
    out_ = linear(input_)
    assert out_.dim() == ((out,), 1)

def test_linear_forward_shape_batched():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    out = random.choice(SIZES)
    in_ = random.choice(SIZES)
    batch_size = random.randint(5, 11)
    linear = Linear(out, in_, pc)
    input_ = [dy.inputVector(np.random.randn(in_)) for _ in range(batch_size)]
    input_ = dy.concatenate_to_batch(input_)
    out_ = linear(input_)
    assert out_.dim() == ((out,), batch_size)

def test_LSTM_shape():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    out = random.choice(SIZES)
    in_ = random.choice(SIZES)
    seq_len = random.randint(5, 11)
    input_ = [dy.inputVector(np.random.randn(in_)) for _ in range(seq_len)]
    lstm = LSTM(out, in_, pc)
    output_ = lstm(input_)
    assert len(output_) == seq_len
    for out_ in output_:
        assert out_.dim() == ((out,), 1)

def test_LSTM_shape_batch():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    out = random.choice(SIZES)
    in_ = random.choice(SIZES)
    seq_len = random.randint(5, 11)
    batch_size = random.randint(5, 11)
    input_ = [dy.concatenate_to_batch([dy.inputVector(np.random.randn(in_)) for _ in range(batch_size)]) for _ in range(seq_len)]
    lstm = LSTM(out, in_, pc)
    output_ = lstm(input_)
    assert len(output_) == seq_len
    for out_ in output_:
        assert out_.dim() == ((out,), batch_size)

def test_LSTM_Encoder_shape():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    out = random.choice(SIZES)
    in_ = random.choice(SIZES)
    seq_len = random.randint(5, 11)
    lens = [seq_len - 1]
    input_ = [dy.inputVector(np.random.randn(in_)) for _ in range(seq_len)]
    lstm = LSTMEncoder(out, in_, pc)
    output_ = lstm(input_, lens)
    assert output_.dim() == ((out,), 1)

def test_LSTM_Encoder_shape_batch():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    out = random.choice(SIZES)
    in_ = random.choice(SIZES)
    seq_len = random.randint(5, 11)
    batch_size = random.randint(5, 11)
    lens = [random.randint(1, seq_len) for _ in range(batch_size)]
    input_ = [dy.concatenate_to_batch([dy.inputVector(np.random.randn(in_)) for _ in range(batch_size)]) for _ in range(seq_len)]
    lstm = LSTMEncoder(out, in_, pc)
    output_ = lstm(input_, lens)
    assert output_.dim() == ((out,), batch_size)

def test_conv_params_shape():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    fsz = random.choice(FILTER_SIZES)
    dsz = random.choice(SIZES)
    cmotsz = random.choice(SIZES)
    conv = Convolution1d(fsz, cmotsz, dsz, pc)
    params = {p.name(): p for p in pc.parameters_list()}
    assert params['/Conv/Weight'].shape() == (1, fsz, dsz, cmotsz)
    assert params['/Conv/Bias'].shape() == (cmotsz,)

def test_conv_output_shape():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    fsz = random.choice(FILTER_SIZES)
    dsz = random.choice(SIZES)
    cmotsz = random.choice(SIZES)
    seq_len = random.randint(fsz + 1, fsz + 5)
    input_ = dy.inputTensor(np.random.randn(1, seq_len, dsz))
    conv = Convolution1d(fsz, cmotsz, dsz, pc)
    output_ = conv(input_)
    assert output_.dim() == ((cmotsz,), 1)

def test_conv_output_shape_batched():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    fsz = random.choice(FILTER_SIZES)
    dsz = random.choice(SIZES)
    cmotsz = random.choice(SIZES)
    seq_len = random.randint(fsz + 1, fsz + 5)
    batch_size = random.randint(5, 11)
    input_ = dy.concatenate_to_batch([dy.inputTensor(np.random.randn(1, seq_len, dsz)) for _ in range(batch_size)])
    conv = Convolution1d(fsz, cmotsz, dsz, pc)
    output_ = conv(input_)
    assert output_.dim() == ((cmotsz,), batch_size)

def test_embedded_dense_shape():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    vsz = random.choice(SIZES)
    dsz = random.choice(SIZES)
    seq_len = random.randint(5, 11)
    input_ = [random.randint(0, vsz) for _ in range(seq_len)]
    embed = Embedding(vsz, dsz, pc, dense=True)
    output_ = embed(input_)
    assert output_.dim() == ((seq_len, dsz), 1)

def test_embedded_shape():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    vsz = random.choice(SIZES)
    dsz = random.choice(SIZES)
    seq_len = random.randint(5, 11)
    input_ = [random.randint(0, vsz) for _ in range(seq_len)]
    embed = Embedding(vsz, dsz, pc)
    output_ = embed(input_)
    assert len(output_) == seq_len
    for out_ in output_:
        assert out_.dim() == ((dsz,), 1)

def test_embedded_shape_batched():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    vsz = random.choice(SIZES)
    dsz = random.choice(SIZES)
    seq_len = random.randint(5, 11)
    batch_size = random.randint(5, 11)
    input_ = [[random.randint(0, vsz) for _ in range(batch_size)] for _ in range(seq_len)]
    embed = Embedding(vsz, dsz, pc, batched=True)
    output_ = embed(input_)
    assert len(output_) == seq_len
    for out_ in output_:
        assert out_.dim() == ((dsz, ), batch_size)

def test_embedded_dense_batched():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    vsz = random.choice(SIZES)
    dsz = random.choice(SIZES)
    seq_len = random.randint(5, 11)
    batch_size = random.randint(5, 11)
    input_ = [[random.randint(0, vsz) for _ in range(batch_size)] for _ in range(seq_len)]
    embed = Embedding(vsz, dsz, pc, dense=True, batched=True)
    output_ = embed(input_)
    output_.dim() == ((dsz, vsz), batch_size)

def test_embedding_from_numpy():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    gold = np.random.randn(200, 100)
    embed = Embedding(12, 12, pc, embedding_weight=gold)
    embedding_weights = pc.lookup_parameters_list()[0]
    np.testing.assert_allclose(gold.T, embedding_weights.npvalue())

def test_embedding_lookup():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    gold = np.random.randn(200, 100)
    embed = Embedding(12, 12, pc, embedding_weight=gold)
    idx = random.randint(0, len(gold))
    vector = embed([idx])
    gold_v = gold[idx, :]
    np.testing.assert_allclose(gold_v, vector[0].npvalue())

def test_embedding_shape():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    vsz = random.choice(SIZES)
    dsz = random.choice(SIZES)
    embed = Embedding(vsz, dsz, pc)
    weights = pc.lookup_parameters_list()[0]
    assert weights.shape() == (vsz, dsz)

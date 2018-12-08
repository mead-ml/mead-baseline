import random
import pytest
import numpy as np
from mock import MagicMock, patch
dy = pytest.importorskip('dynet')
import baseline
from baseline.dy.dynety import *


SIZES = [128, 256, 300, 512]
FILTER_SIZES = [3, 4, 5]


def setup_function(function):
    dy.renew_cg()


class M():
    def __init__(self, pc): self.pc = pc


def test_linear_params_present():
    pc = dy.ParameterCollection()
    linear = Linear(12, 6, pc)
    names = {p.name() for p in pc.parameters_list()}
    assert "/linear/weight" in names
    assert "/linear/bias" in names

def test_linear_params_rename():
    pc = dy.ParameterCollection()
    gold = "TESTING"
    linear = Linear(12, 6, pc, name=gold)
    names = {p.name() for p in pc.parameters_list()}
    assert "/{}/weight".format(gold) in names
    assert "/{}/bias".format(gold) in names

def test_linear_param_shapes():
    pc = dy.ParameterCollection()
    out = random.choice(SIZES)
    in_ = random.choice(SIZES)
    linear = Linear(out, in_, pc)
    params = {p.name(): p for p in pc.parameters_list()}
    assert params['/linear/weight'].shape() == (out, in_)
    assert params['/linear/bias'].shape() == (out,)

def test_linear_forward_shape():
    pc = dy.ParameterCollection()
    out = random.choice(SIZES)
    in_ = random.choice(SIZES)
    linear = Linear(out, in_, pc)
    input_ = dy.inputVector(np.random.randn(in_))
    out_ = linear(input_)
    assert out_.dim() == ((out,), 1)

def test_linear_forward_shape_batched():
    pc = dy.ParameterCollection()
    out = random.choice(SIZES)
    in_ = random.choice(SIZES)
    batch_size = random.randint(5, 11)
    linear = Linear(out, in_, pc)
    input_ = [dy.inputVector(np.random.randn(in_)) for _ in range(batch_size)]
    input_ = dy.concatenate_to_batch(input_)
    out_ = linear(input_)
    assert out_.dim() == ((out,), batch_size)

def test_rnn_forward_shape():
    pc = dy.ParameterCollection()
    out = random.choice(SIZES)
    in_ = random.choice(SIZES)
    batch_size = random.randint(1, 6)
    seq_len = random.randint(5, 11)
    input_ = [dy.inputTensor(np.random.randn(in_, batch_size), True) for _ in range(seq_len)]
    layers = random.randint(1, 4)
    rnn = dy.LSTMBuilder(layers, in_, out, pc)
    output_ = rnn_forward(rnn, input_)
    assert len(output_) == seq_len
    for out_ in output_:
        assert out_.dim() == ((out,), batch_size)

def test_rnn_forward_birnn():
    pc = dy.ParameterCollection()
    out = random.choice(SIZES)
    in_ = random.choice(SIZES)
    batch_size = random.randint(1, 6)
    seq_len = random.randint(5, 11)
    input_ = [dy.inputTensor(np.random.randn(in_, batch_size), True) for _ in range(seq_len)]
    layers = random.randint(1, 4)
    rnn = dy.BiRNNBuilder(layers, in_, out, pc, dy.VanillaLSTMBuilder)
    output_ = rnn_forward(rnn, input_)
    assert len(output_) == seq_len
    for out_ in output_:
        assert out_.dim() == ((out,), batch_size)

def test_rnn_with_state():
    pc = dy.ParameterCollection()
    out = random.choice(SIZES)
    in_ = random.choice(SIZES)
    batch_size = random.randint(1, 6)
    seq_len = random.randint(5, 11)
    input_ = [dy.inputTensor(np.random.randn(in_, batch_size), True) for _ in range(seq_len)]
    layers = random.randint(1, 4)
    rnn = dy.LSTMBuilder(layers, in_, out, pc)
    output_, state = rnn_forward_with_state(rnn, input_)
    assert len(output_) == seq_len
    for out_ in output_:
        assert out_.dim() == ((out,), batch_size)
    assert len(state) == layers * 2
    for s in state:
        assert s.dim() == ((out,), batch_size)

def test_rnn_with_state_with_prev():
    seq_len = np.random.randint(5, 10)
    batch_size = np.random.randint(2, 5)
    inputs = np.random.rand(seq_len, 100, batch_size)
    dy.renew_cg()
    pc = dy.ParameterCollection()
    lstm = dy.VanillaLSTMBuilder(1, 100, 100, pc)
    o, l = rnn_forward_with_state(lstm, [dy.inputTensor(in_, True) for in_ in inputs])
    np_o = [x.npvalue() for x in o]
    np_l = [x.npvalue() for x in l]
    dy.renew_cg()
    o, l = rnn_forward_with_state(lstm, [dy.inputTensor(in_, True) for in_ in inputs], state=np_l)
    np_o2 = [x.npvalue() for x in o]
    for o, o2 in zip(np_o, np_o2):
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(o, o2)

def test_rnn_with_state_full_len_matches_ends():
    pc = dy.ParameterCollection()
    out = random.choice(SIZES)
    in_ = random.choice(SIZES)
    batch_size = random.randint(1, 6)
    seq_len = random.randint(5, 11)
    input_ = [dy.inputTensor(np.random.randn(in_, batch_size), True) for _ in range(seq_len)]
    layers = random.randint(1, 4)
    rnn = dy.LSTMBuilder(layers, in_, out, pc)
    output_, state = rnn_forward_with_state(rnn, input_)
    output_2, state_2 = rnn_forward_with_state(rnn, input_, lengths=[seq_len] * batch_size)
    for s1, s2 in zip(state, state_2):
        np.testing.assert_allclose(s1.npvalue(), s2.npvalue())

def test_conv_params_shape():
    pc = dy.ParameterCollection()
    fsz = random.choice(FILTER_SIZES)
    dsz = random.choice(SIZES)
    cmotsz = random.choice(SIZES)
    conv = Convolution1d(fsz, cmotsz, dsz, pc)
    params = {p.name(): p for p in pc.parameters_list()}
    assert params['/conv/weight'].shape() == (1, fsz, dsz, cmotsz)
    assert params['/conv/bias'].shape() == (cmotsz,)

def test_conv_output_shape():
    pc = dy.ParameterCollection()
    fsz = random.choice(FILTER_SIZES)
    dsz = random.choice(SIZES)
    cmotsz = random.choice(SIZES)
    seq_len = random.randint(fsz + 1, fsz + 5)
    input_ = dy.inputTensor(np.random.randn(1, seq_len, dsz))
    conv = Convolution1d(fsz, cmotsz, dsz, pc)
    output_ = conv(input_)
    assert output_.dim() == ((1, seq_len, cmotsz,), 1)

def test_conv_output_shape_batched():
    pc = dy.ParameterCollection()
    fsz = random.choice(FILTER_SIZES)
    dsz = random.choice(SIZES)
    cmotsz = random.choice(SIZES)
    seq_len = random.randint(fsz + 1, fsz + 5)
    batch_size = random.randint(5, 11)
    input_ = dy.concatenate_to_batch([dy.inputTensor(np.random.randn(1, seq_len, dsz)) for _ in range(batch_size)])
    conv = Convolution1d(fsz, cmotsz, dsz, pc)
    output_ = conv(input_)
    assert output_.dim() == ((1, seq_len, cmotsz,), batch_size)

def test_conv_parameter_init_glorot():
    pc = dy.ParameterCollection()
    fsz = random.choice(FILTER_SIZES)
    dsz = random.choice(SIZES)
    cmotsz = random.choice(SIZES)
    conv = Convolution1d(fsz, cmotsz, dsz, pc)
    conv_weight = pc.parameters_list()[0]
    gold = 0.5 * np.sqrt(6.0 / (fsz * dsz + fsz * cmotsz))
    min_ = np.min(conv_weight.as_array())
    max_ = np.max(conv_weight.as_array())
    np.testing.assert_allclose(min_, -gold, atol=1e-5)
    np.testing.assert_allclose(max_, gold, atol=1e-5)


@pytest.fixture
def matmul_dims():
    D = np.random.randint(2, 10)
    T = np.random.randint(5, 20)
    D2 = np.random.randint(15, 30)
    H = np.random.randint(5, 10, size=np.random.randint(1, 3))
    B = np.random.randint(2, 9)
    return D, T, D2, H, B


def test_matmul_shape(matmul_dims):
    D, T, D2, H, B = matmul_dims
    x_dim = tuple(np.hstack([D, T, H]))
    y_dim = tuple(np.hstack([T, D2, H]))
    gold = (tuple(np.hstack([D, D2, H])), B)
    x = dy.random_normal(x_dim, -1, 1, batch_size=B)
    y = dy.random_normal(y_dim, -1, 1, batch_size=B)
    z = batch_matmul(x, y)
    assert z.dim() == gold


def test_sequence_mask_shape():
    B = np.random.randint(5, 10)
    T = np.random.randint(1, 20, size=B)
    max_T = np.max(T)
    mask = sequence_mask(T)[0]
    gold = ((max_T, 1), B)
    assert mask.dim() == gold


def test_sequence_mask_max_shape():
    B = np.random.randint(5, 10)
    T = np.random.randint(1, 20, size=B)
    max_T = np.random.randint(22, 30)
    mask = sequence_mask(T, max_T)[0]
    gold = ((max_T, 1), B)
    assert mask.dim() == gold


def test_sequence_mask_max_shape_less():
    B = np.random.randint(5, 10)
    T = np.random.randint(10, 20, size=B)
    max_T = np.random.randint(2, 10)
    mask = sequence_mask(T, max_T)[0]
    gold = ((max_T, 1), B)
    assert mask.dim() == gold


def test_sequence_mask_valid_count():
    B = np.random.randint(5, 10)
    T = np.random.randint(1, 20, size=B)
    mask = sequence_mask(T)[0].npvalue()
    gold = np.sum(T)
    assert mask.sum() == gold


def test_sequence_mask_valid_loc():
    B = np.random.randint(5, 10)
    lens = np.random.randint(1, 20, size=B)
    mask = sequence_mask(lens)[0].npvalue().squeeze()
    max_T = mask.shape[0]

    def test(mask, lens, T, B):
        t = np.random.randint(0, T)
        b = np.random.randint(0, B)
        if t < lens[b]:
            assert mask[t, b] == 1
        else:
            assert mask[t, b] == 0

    for _ in range(100):
        test(mask, lens, max_T, B)


def test_folded_softmax():
    H, T, X, B = np.random.randint(1, 10, size=4)
    in_ = dy.inputTensor(np.random.rand(H, T, X, B), batched=True)
    out = folded_softmax(in_)
    golds = [dy.softmax(dy.pick(in_, i, dim=2), d=0) for i in range(X)]
    gold = np.concatenate([np.expand_dims(g.npvalue(), 2) for g in golds], axis=2)
    np.testing.assert_allclose(out.npvalue(), gold)


def test_squeeze():
    dims = (1, 32, 45, 1)
    gold = (32, 45)
    in_ = dy.random_normal(dims)
    out = squeeze(in_)
    assert out.dim()[0] == gold


def test_squeeze_right_number():
    ndims = np.random.randint(2, 4)
    dims = np.random.randint(1, 10, size=ndims)
    gold = np.count_nonzero(dims != 1)
    in_ = dy.random_normal(tuple(dims))
    out = squeeze(in_)
    assert len(out.dim()[0]) == gold


def test_squeeze_dim():
    ndims = np.random.randint(2, 4)
    dims = np.random.randint(2, 10, size=ndims)
    single = np.random.randint(0, ndims)
    dims[single] = 1
    gold = tuple(x for i, x in enumerate(dims) if i != single)
    in_ = dy.random_normal(tuple(dims))
    res = squeeze(in_, single)
    assert res.dim()[0] == gold


def test_squeeze_invalid_dim():
    ndims = np.random.randint(2, 4)
    dims = np.random.randint(2, 10, size=ndims)
    dim = np.random.randint(0, ndims)
    in_ = dy.random_normal(tuple(dims))
    with pytest.raises(AssertionError):
        squeeze(in_, dim)


def test_unsqueeze_right_number():
    ndims = np.random.randint(2, 4)
    dims = np.random.randint(1, 10, size=ndims)
    gold = ndims + 1
    in_ = dy.random_normal(tuple(dims))
    out = unsqueeze(in_, 0)
    assert len(out.dim()[0]) == gold


def test_unsqueeze_neg():
    ndims = np.random.randint(2, 4)
    dims = np.random.randint(2, 10, size=ndims)

    def test(dims, ndims):
        d = np.random.randint(-ndims, 0)
        in_ = dy.random_normal(tuple(dims))
        out = unsqueeze(in_, d)
        assert out.dim()[0][d] == 1

    for _ in range(100):
        test(dims, ndims)


def test_unsqueeze_pos():
    ndims = np.random.randint(2, 4)
    dims = np.random.randint(2, 10, size=ndims)

    def test(dims, ndims):
        d = np.random.randint(0, ndims)
        in_ = dy.random_normal(tuple(dims))
        out = unsqueeze(in_, d)
        assert out.dim()[0][d] == 1

    for _ in range(100):
        test(dims, ndims)


def test_unsqueeze():
    dims = (4, 4, 4)
    gold = (4, 1, 4, 4)
    in_ = dy.random_normal(dims)
    out = unsqueeze(in_, 1)
    assert out.dim()[0] == gold


def test_transpose():
    dims = (2, 3, 4, 5)
    gold = (3, 2, 4, 5)
    swap = [0, 1]
    in_ = dy.random_normal(dims)
    out = transpose(in_, *swap)
    assert out.dim()[0] == gold


def test_transpose_negatives():
    dims = (2, 3, 4, 5)
    gold = (2, 3, 5, 4)
    swap = [-2, -1]
    in_ = dy.random_normal(dims)
    out = transpose(in_, *swap)
    assert out.dim()[0] == gold


def test_transpose_both():
    B, T, H = 50, 10, 8
    dims = (B, T, H)
    gold = (H, T, B)
    in_ = dy.zeros(dims)
    res = transpose(in_, 0, -1)
    assert res.dim()[0] == gold

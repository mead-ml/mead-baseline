import random
import pytest
import numpy as np
from mock import MagicMock, patch
dy = pytest.importorskip('dynet')
import baseline
from baseline.dy.dynety import *

SIZES = [128, 256, 300, 512]
FILTER_SIZES = [3, 4, 5]

class M():
    def __init__(self, pc): self.pc = pc

def test_optimizer_adadelta():
    dy.renew_cg()
    m = M(dy.ParameterCollection())
    opt = optimizer(m, 'adadelta')
    assert isinstance(opt, dy.AdadeltaTrainer)

# def test_folded_linear():
#     dy.renew_cg()
#     dummy_weight = np.random.rand(100, 50)
#     dummy_bias = np.random.rand(100)
#     pc_mock = MagicMock()
#     pc_mock.add_subcollection.return_value = pc_mock
#     def dummy(shape, *args, **kwargs):
#         if isinstance(shape, tuple):
#             return dy.inputTensor(dummy_weight)
#         return dy.inputTensor(dummy_bias)

#     pc_mock.add_parameters.side_effect = dummy
#     l = Linear(100, 50, pc_mock)
#     fl = FoldedLinear(100, 50, pc_mock)

#     input_ = np.random.rand(8, 50)
#     ls = [l(dy.inputTensor(x)) for x in input_]
#     fls = fl(dy.inputTensor(input_))
#     res1 = [x.npvalue() for x in ls]
#     res1 = np.stack(res1)
#     res2 = fls.npvalue()
#     np.testing.assert_allclose(res1, res2)


def test_optimizer_adam():
    dy.renew_cg()
    m = M(dy.ParameterCollection())
    opt = optimizer(m, 'adam')
    assert isinstance(opt, dy.AdamTrainer)

def test_optimizer_rmsprop():
    dy.renew_cg()
    gold_lr = 5.0
    m = M(dy.ParameterCollection())
    opt = optimizer(m, 'rmsprop', eta=gold_lr)
    assert isinstance(opt, dy.RMSPropTrainer)
    assert opt.learning_rate == gold_lr

def test_optimizer_sgd():
    dy.renew_cg()
    gold_lr = 5.0
    m = M(dy.ParameterCollection())
    opt = optimizer(m, 'sgd', eta=gold_lr, mom=0.0)
    assert isinstance(opt, dy.SimpleSGDTrainer)
    assert opt.learning_rate == gold_lr

def test_optimizer_momentum_sgd():
    dy.renew_cg()
    gold_lr = 5.0
    gold_mom = 0.1
    m = M(dy.ParameterCollection())
    opt = optimizer(m, 'sgd', eta=gold_lr, mom=gold_mom)
    assert isinstance(opt, dy.MomentumSGDTrainer)
    assert opt.learning_rate == gold_lr
    # Dynet doesn't expose the mom value in the trainer
    # assert opt.mom == gold_mom

def test_optimizer_lr_works_too():
    dy.renew_cg()
    gold_lr = 5.0
    m = M(dy.ParameterCollection())
    opt = optimizer(m, 'sgd', lr=gold_lr, mom=0.0)
    assert isinstance(opt, dy.SimpleSGDTrainer)
    assert opt.learning_rate == gold_lr

def test_optimizer_clip_not_called():
    dy.renew_cg()
    m = M(dy.ParameterCollection())
    with patch('baseline.dy.dynety.dy.AdamTrainer') as opt_mock:
        opt_mock.return_value = MagicMock()
        opt = optimizer(m, 'adam')
        assert opt.set_clip_threshold.call_count == 0

def test_optimizer_clip_not_called():
    dy.renew_cg()
    m = M(dy.ParameterCollection())
    with patch('baseline.dy.dynety.dy.AdamTrainer') as opt_mock:
        opt_mock.return_value = MagicMock()
        opt = optimizer(m, 'adam', clip=5.0)
        assert opt.set_clip_threshold.call_count == 1

def test_linear_params_present():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    linear = Linear(12, 6, pc)
    names = {p.name() for p in pc.parameters_list()}
    assert "/linear/weight" in names
    assert "/linear/bias" in names

def test_linear_params_rename():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    gold = "TESTING"
    linear = Linear(12, 6, pc, name=gold)
    names = {p.name() for p in pc.parameters_list()}
    assert "/{}/weight".format(gold) in names
    assert "/{}/bias".format(gold) in names

def test_linear_param_shapes():
    dy.renew_cg()
    pc = dy.ParameterCollection()
    out = random.choice(SIZES)
    in_ = random.choice(SIZES)
    linear = Linear(out, in_, pc)
    params = {p.name(): p for p in pc.parameters_list()}
    assert params['/linear/weight'].shape() == (out, in_)
    assert params['/linear/bias'].shape() == (out,)

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

def test_rnn_forward_shape():
    dy.renew_cg()
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
    dy.renew_cg()
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
    dy.renew_cg()
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
    dy.renew_cg()
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
    dy.renew_cg()
    pc = dy.ParameterCollection()
    fsz = random.choice(FILTER_SIZES)
    dsz = random.choice(SIZES)
    cmotsz = random.choice(SIZES)
    conv = Convolution1d(fsz, cmotsz, dsz, pc)
    params = {p.name(): p for p in pc.parameters_list()}
    assert params['/conv/weight'].shape() == (1, fsz, dsz, cmotsz)
    assert params['/conv/bias'].shape() == (cmotsz,)

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

def test_conv_parameter_init_glorot():
    dy.renew_cg()
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

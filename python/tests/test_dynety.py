import random
import pytest
import numpy as np
dy = pytest.importorskip('dynet')
from mock import MagicMock, patch
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

def test_truncated_lstm_output_shapes():
    seq_len = np.random.randint(5, 10)
    dy.renew_cg()
    pc = dy.ParameterCollection()
    lstm = TruncatedLSTM(100, 100, pc)
    o, _ = lstm([dy.inputVector(np.random.rand(100)) for _ in range(seq_len)])
    assert len(o) == seq_len
    assert o[0].dim() == ((100,), 1)

def test_truncated_lstm_state_shapes():
    seq_len = np.random.randint(5, 10)
    layers = np.random.randint(1, 3)
    dy.renew_cg()
    pc = dy.ParameterCollection()
    lstm = TruncatedLSTM(100, 100, pc, layers)
    _, l = lstm([dy.inputVector(np.random.rand(100)) for _ in range(seq_len)])
    assert len(l) == 2 * layers
    assert l[0].dim() == ((100,), 1)

def test_truncated_lstm_input_state_changes_things():
    seq_len = np.random.randint(5, 10)
    inputs = np.random.rand(seq_len, 100)
    dy.renew_cg()
    pc = dy.ParameterCollection()
    lstm = TruncatedLSTM(100, 100, pc, 1, batched=False)
    o, l = lstm([dy.inputVector(in_) for in_ in inputs])
    np_o = [x.npvalue() for x in o]
    np_l = [x.npvalue() for x in l]
    dy.renew_cg()
    o, l = lstm([dy.inputVector(in_) for in_ in inputs], np_l)
    np_o2 = [x.npvalue() for x in o]
    for o, o2 in zip(np_o, np_o2):
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(o, o2)

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
    gold = 0.5 * np.sqrt(6 / (fsz * dsz + fsz * cmotsz))
    min_ = np.min(conv_weight.as_array())
    max_ = np.max(conv_weight.as_array())
    np.testing.assert_allclose(min_, -gold, atol=1e-5)
    np.testing.assert_allclose(max_, gold, atol=1e-5)

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

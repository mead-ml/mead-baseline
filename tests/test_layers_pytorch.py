import pytest
import numpy as np

torch = pytest.importorskip("torch")
from eight_mile.utils import Offsets
from eight_mile.pytorch.layers import *

C = 10
B = 50
T = 20
H = 8
L = 2
TRIALS = 100


@pytest.fixture
def lengths():
    lengths = torch.randint(1, T, size=(B,)).long()
    return lengths


@pytest.fixture
def logits(lengths):
    logits = torch.rand(B, T, C)
    for i, l in enumerate(lengths):
        logits[i, l:, :] = 0
    return logits


@pytest.fixture
def labels(lengths):
    lab = torch.randint(1, C, size=(B, T)).long()
    for i, l in enumerate(lengths):
        lab[i, l:] = 0
    return lab

@pytest.fixture
def encoder_input(lengths):
    hidden = torch.rand(B, T, C)
    lengths, perm_idx = lengths.sort(0, descending=True)
    return hidden[perm_idx], lengths

def generate_data_with_zeros(lengths):
    data = torch.randint(1, 100, (len(lengths), torch.max(lengths)))
    for i, length in enumerate(lengths):
        data[i, length:] = 0
        if length // 2 > 0:
            extra_zeros = torch.randint(0, length.item() - 1, size=((length // 2).item(),))
            data[i, extra_zeros] = 0
    return data


def test_infer_lengths(lengths):
    def test():
        data = generate_data_with_zeros(lengths)
        infered = infer_lengths(data, dim=1)
        np.testing.assert_equal(infered.numpy(), lengths.numpy())

    for _ in range(TRIALS):
        test()


def test_infer_lengths_t_first(lengths):
    def test():
        data = generate_data_with_zeros(lengths)
        data = data.transpose(0, 1)
        infered = infer_lengths(data, dim=0)
        np.testing.assert_equal(infered.numpy(), lengths.numpy())

    for _ in range(TRIALS):
        test()


def test_infer_legnths_multi_dim():
    data = torch.rand(10, 11, 12)
    with pytest.raises(ValueError):
        infer_lengths(data)


def raw_loss(logits, labels, loss):
    B, T, H = logits.size()
    crit = loss(reduce=False, ignore_index=Offsets.PAD)
    total_size = labels.nelement()
    res = crit(logits.view(total_size, -1), labels.view(total_size))
    return res.view(B, T)


def test_batch_sequence_loss(logits, labels):
    loss = torch.nn.CrossEntropyLoss
    raw = raw_loss(logits, labels, loss)
    gold = torch.mean(torch.sum(raw, dim=1))
    crit = SequenceLoss(LossFn=loss, avg="batch")
    res = crit(logits, labels)
    np.testing.assert_allclose(res.numpy(), gold.numpy(), rtol=1e-6)


def test_token_sequence_loss(logits, labels, lengths):
    loss = torch.nn.CrossEntropyLoss
    raw = raw_loss(logits, labels, loss)
    gold = torch.sum(raw) / torch.sum(lengths).to(logits.dtype)
    crit = SequenceLoss(LossFn=loss, avg="token")
    res = crit(logits, labels)
    np.testing.assert_allclose(res.numpy(), gold.numpy(), rtol=1e-6)


def test_gru_sequence_shapes(encoder_input):

    z = GRUEncoderSequence(C, H, L, batch_first=True)
    out = z(encoder_input)
    seq_len = torch.max(encoder_input[1]).item()
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

    seq_len = torch.max(encoder_input[1]).item()
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
    seq_len = torch.max(encoder_input[1]).item()
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



def test_lstm_shc_shapes(encoder_input):

    seq_len = torch.max(encoder_input[1]).item()
    z = LSTMEncoderSequenceHiddenContext(C, H, L, batch_first=True)
    seq, (s1, s2) = z(encoder_input)
    np.testing.assert_equal(seq.shape, (B, seq_len, H))
    np.testing.assert_equal(s1.shape, (B, H))
    np.testing.assert_equal(s2.shape, (B, H))

    z = BiLSTMEncoderSequenceHiddenContext(C, H, L, batch_first=True)
    seq, (s1, s2) = z(encoder_input)
    np.testing.assert_equal(seq.shape, (B, seq_len, H))
    np.testing.assert_equal(s1.shape, (B, H))
    np.testing.assert_equal(s2.shape, (B, H))


def test_lstm_all_shapes(encoder_input):

    seq_len = torch.max(encoder_input[1]).item()
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

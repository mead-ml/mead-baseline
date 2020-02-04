import pytest

torch = pytest.importorskip("torch")
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from eight_mile.pytorch.layers import sequence_mask, subsequent_mask


@pytest.fixture
def lengths():
    batch_size = np.random.randint(5, 10)
    max_seq = np.random.randint(15, 20)
    lengths = torch.LongTensor(np.random.randint(1, max_seq, size=[batch_size]))
    seq_len = torch.max(lengths).item()
    return batch_size, lengths, seq_len


def test_mask_shape(lengths):
    bsz, lengths, seq_len = lengths
    mask = sequence_mask(lengths)
    assert mask.size(0) == bsz
    assert mask.size(1) == seq_len


def test_mask_valid_locs(lengths):
    bsz, lengths, seq_len = lengths
    mask = sequence_mask(lengths)
    np_mask = np.zeros((bsz, seq_len))
    for i in range(bsz):
        for j in range(seq_len):
            if j < lengths.data[i]:
                np_mask[i, j] = 1
    np.testing.assert_allclose(mask.data.numpy(), np_mask)


def test_mask_mxlen(lengths):
    bsz, lengths, seq_len = lengths
    extra = np.random.randint(2, 11)
    mask = sequence_mask(lengths, seq_len + extra)
    np_mask = np.zeros((bsz, seq_len + extra))
    for i in range(bsz):
        for j in range(seq_len + extra):
            if j < lengths.data[i]:
                np_mask[i, j] = 1
    np.testing.assert_allclose(mask.data.numpy(), np_mask)


def test_attention_masked_valid_probs(lengths):
    bsz, lengths, seq_len = lengths
    mask = sequence_mask(lengths)
    scores = torch.rand(bsz, seq_len)
    score_mask = scores.masked_fill(mask, -1e9)
    attention_weights = F.softmax(score_mask, dim=1)
    for row in attention_weights:
        np.testing.assert_allclose(torch.sum(row).numpy(), 1.0, rtol=1e-5)


def test_attention_masked_ignores_pad(lengths):
    bsz, lengths, seq_len = lengths
    mask = sequence_mask(lengths)
    scores = torch.rand(bsz, seq_len)
    score_mask = scores.masked_fill(mask, -1e9)
    attention_weights = F.softmax(score_mask, dim=1)
    for row, length in zip(attention_weights, lengths):
        if length.item() == seq_len:
            continue
        masked = row[: length.item()]
        np.testing.assert_allclose(masked.data.numpy(), 0.0)


def test_seq_mask_valid_count(lengths):
    bsz, lengths, _ = lengths
    mask = sequence_mask(lengths)
    gold = lengths.sum()
    assert mask.sum() == gold.sum()


def test_subsequent_mask_shape():
    T = np.random.randint(2, 50)
    gold = (1, 1, T, T)
    mask = subsequent_mask(T)
    assert mask.shape == gold


def test_subsequent_mask_valid_count():
    T = np.random.randint(4, 50)
    gold = (T * (T + 1)) / 2
    mask = subsequent_mask(T).numpy()
    assert np.sum(mask) == gold


def test_subsequent_mask_valid_loc():
    T = np.random.randint(4, 100)
    mask = subsequent_mask(T).numpy().squeeze()

    def test(T, mask):
        i, j = np.random.randint(0, T, size=2)
        if i < j:
            assert mask[i, j] == 0
        else:
            assert mask[i, j] == 1

    for _ in range(100):
        test(T, mask)

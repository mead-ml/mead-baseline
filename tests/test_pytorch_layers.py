import pytest
import numpy as np

torch = pytest.importorskip("torch")
from eight_mile.utils import Offsets
from eight_mile.pytorch.layers import SequenceLoss

C = 10
B = 50
S = 20


@pytest.fixture
def lengths():
    lengths = torch.randint(1, S, size=(B,)).long()
    return lengths


@pytest.fixture
def logits(lengths):
    logits = torch.rand(B, S, C)
    for i, l in enumerate(lengths):
        logits[i, l:, :] = 0
    return logits


@pytest.fixture
def labels(lengths):
    lab = torch.randint(1, C, size=(B, S)).long()
    for i, l in enumerate(lengths):
        lab[i, l:] = 0
    return lab


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

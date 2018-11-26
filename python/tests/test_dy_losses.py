import pytest
import numpy as np
dy = pytest.importorskip('dynet')


@pytest.fixture
def shapes():
    B = np.random.randint(20, 51)
    S = np.random.randint(30, 61)
    C = np.random.randint(10, 100)
    return B, S, C


@pytest.fixture
def logits(shapes):
    return np.random.rand(*shapes)


@pytest.fixture
def labels(shapes, lengths):
    B, S, C = shapes
    labels = np.random.randint(1, C, size=(B, S))
    for i, l in enumerate(lengths):
        labels[i, l:] = 0
    return labels


@pytest.fixture
def lengths(shapes):
    B, S, _ = shapes
    lengths = np.random.randint(1, S, size=B)
    lengths[np.random.choice(np.arange(B), replace=False, size=B//2)] = S
    return lengths


def test_masked_token_level_loss(shapes, logits, labels, lengths):
    from baseline.dy.seq2seq.train import Seq2SeqTrainerDynet
    dy.renew_cg()
    B, S, C = shapes
    dy_logits = [dy.inputTensor(x, batched=True) for x in logits.transpose(1, 2, 0)]
    dy_labels = labels.transpose(1, 0)
    gold = dy.zeros((1,))
    for b in range(B):
        for i, (logit, label) in enumerate(zip(dy_logits, dy_labels)):
            if i >= lengths[b]:
                continue
            log = dy.pick_batch_elem(logit, b)
            lab = label[b]
            gold += dy.pickneglogsoftmax(log, lab)
    gold = gold.npvalue() / np.sum(lengths)

    res = Seq2SeqTrainerDynet._loss(dy_logits, dy_labels, lengths)
    np.testing.assert_allclose(res.npvalue(), gold, rtol=1e-6)


def test_token_level_loss(shapes, logits, labels, lengths):
    from baseline.dy.lm.train import LanguageModelTrainerDynet
    dy.renew_cg()
    B, S, C = shapes
    dy_logits = [dy.inputTensor(x, batched=True) for x in logits.transpose(1, 2, 0)]
    dy_labels = labels.transpose(1, 0)
    gold = dy.zeros((1,))
    for b in range(B):
        for logit, label in zip(dy_logits, dy_labels):
            log = dy.pick_batch_elem(logit, b)
            lab = label[b]
            gold += dy.pickneglogsoftmax(log, lab)
    gold = gold.npvalue() / (B * S)

    res = LanguageModelTrainerDynet._loss(dy_logits, dy_labels)
    np.testing.assert_allclose(res.npvalue(), gold, rtol=1e-6)

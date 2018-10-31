import pytest
from mock import patch, MagicMock
import numpy as np
import dynet as dy
from baseline.dy.dynety import sequence_mask
from baseline.dy.transformer import subsequent_mask
from baseline.dy.transformer import MultiHeadedAttention as MHA
from baseline.dy.transformer import scaled_dot_product_attention as sdpa
from baseline.dy.transformer import dot_product_attention as dpa


def set_function(function):
    dy.renew_cg()


@pytest.fixture
def qkv():
    D, T, H, B = np.random.randint(5, 10, size=4)
    q = dy.random_normal((D, T, H), batch_size=B)
    k = dy.random_normal((D, T, H), batch_size=B)
    v = dy.random_normal((D, T, H), batch_size=B)
    return q, k, v


def test_subsequent_mask_shape():
    T = np.random.randint(2, 50)
    gold = ((T, T, 1), 1)
    masks = subsequent_mask(T)
    for mask in masks:
        assert mask.dim() == gold


def test_subsequent_mask_valid_count():
    T = np.random.randint(4, 50)
    gold = (T * (T + 1)) / 2
    masks = subsequent_mask(T)
    mask = masks[0].npvalue()
    assert np.sum(mask) == gold


def test_subsequent_mask_valid_loc():
    T = np.random.randint(4, 100)
    mask = subsequent_mask(T)[0].npvalue().squeeze()

    def test(T, mask):
        i, j = np.random.randint(0, T, size=2)
        if i > j:
            assert mask[i, j] == 0
        else:
            assert mask[i, j] == 1

    for _ in range(100):
        test(T, mask)


def test_sdpa_attn(qkv):
    attn_shapes(sdpa, qkv)


def test_dpa_attn(qkv):
    attn_shapes(dpa, qkv)


def attn_shapes(attn, qkv):
    q, k, v = qkv
    gold_dim = q.dim()
    x = attn(q, k, v)
    assert x.dim() == gold_dim


def test_sdpa_applies_dropout(qkv):
    attn_applies_dropout(sdpa, qkv)


def test_dpa_applies_dropout(qkv):
    attn_applies_dropout(dpa, qkv)


def attn_applies_dropout(attn, qkv):
    q, k, v = qkv
    ((_, T, H), B) = q.dim()
    w = dy.inputTensor(np.random.rand(T, T, H, B), batched=True)
    pdrop = np.random.uniform(0.1, 0.9)
    with patch('baseline.dy.transformer.dy.dropout') as d_mock:
        with patch('baseline.dy.transformer.folded_softmax') as fs_mock:
            d_mock.side_effect = lambda x, y: x
            fs_mock.return_value = w
            _ = attn(q, k, v)
            d_mock.assert_not_called()
            _ = attn(q, k, v, dropout=pdrop)
            d_mock.assert_called_once_with(w, pdrop)


def test_sdpa_values(qkv):
    attn_values(sdpa, qkv)


def test_dpa_values(qkv):
    attn_values(dpa, qkv)


def attn_values(attn, qkv):
    q, k, v = qkv
    ((_, T, H), B) = q.dim()
    q = dy.zeros(q.dim()[0], batch_size=q.dim()[1])
    res = attn(q, k, v)
    res = res.npvalue()
    gold = v.npvalue()
    for b in range(B):
        for h in range(H):
            for t in range(T):
                np.testing.assert_allclose(res[:, t, h, b], np.mean(gold, axis=1)[:, h, b], atol=1e-5)


def test_sdpa_values_seq_mask(qkv):
    attn_values_seq_mask(sdpa, qkv)


def test_dpa_values_seq_mask(qkv):
    attn_values_seq_mask(dpa, qkv)


def attn_values_seq_mask(attn, qkv):
    q, k, v = qkv
    ((_, T, H), B) = q.dim()
    q = dy.zeros(q.dim()[0], batch_size=q.dim()[1])
    lens = np.random.randint(1, T, size=B)
    mask = sequence_mask(lens, T)
    res = attn(q, k, v, mask=mask).npvalue()
    gold = v.npvalue()
    for b in range(B):
        for h in range(H):
            for t in range(T):
                np.testing.assert_allclose(res[:, t, h, b], np.mean(gold[:, :lens[b], :, :], axis=1)[:, h, b], atol=1e-5)


def test_sdpa_values_sub_mask(qkv):
    attn_values_sub_mask(sdpa, qkv)


def test_dpa_values_sub_mask(qkv):
    attn_values_sub_mask(dpa, qkv)


def attn_values_sub_mask(attn, qkv):
    q, k, v = qkv
    ((_, T, H), B) = q.dim()
    q = dy.zeros(q.dim()[0], batch_size=q.dim()[1])
    mask = subsequent_mask(T)
    res = attn(q, k, v, mask=mask).npvalue()
    gold = v.npvalue()
    for b in range(B):
        for h in range(H):
            for t in range(T):
                np.testing.assert_allclose(res[:, t, h, b], np.mean(gold[:, :t+1, :, :], axis=1)[:, h, b], atol=1e-5)

def test_multi_headed_attn_assert():
    with pytest.raises(AssertionError):
        d_model = np.random.choice([128, 256, 512])
        h = 1
        while (d_model % h == 0):
            h = np.random.randint(4, 8)
        mha = MHA(h, d_model, None, MagicMock())

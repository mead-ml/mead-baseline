import pytest
import numpy as np
import torch
from baseline.pytorch.torchy import sequence_mask
from baseline.pytorch.transformer import subsequent_mask
from baseline.pytorch.transformer import scaled_dot_product_attention as sdpa
from baseline.pytorch.transformer import dot_product_attention as dpa


@pytest.fixture
def qkv():
    B, H, T, D = map(int, np.random.randint(5, 10, size=4))
    q = torch.rand(B, H, T, D)
    k = torch.rand(B, H, T, D)
    v = torch.rand(B, H, T, D)
    return q, k, v


def test_sdpa_values(qkv):
    attn_values(sdpa, qkv)


def test_dpa_values(qkv):
    attn_values(dpa, qkv)


def attn_values(attn, qkv):
    q, k, v = qkv
    B, H, T, _ = q.shape
    q = q.zero_()
    res, _ = attn(q, k, v)
    res = res.numpy()
    gold = v.numpy()
    for b in range(B):
        for h in range(H):
            for t in range(T):
                np.testing.assert_allclose(res[b, h, t, :], np.mean(gold, axis=2)[b, h, :], rtol=1e-5)


def test_sdpa_values_seq_mask(qkv):
    attn_values_seq_mask(sdpa, qkv)


def test_dpa_values_seq_mask(qkv):
    attn_values_seq_mask(dpa, qkv)


def attn_values_seq_mask(attn, qkv):
    q, k, v = qkv
    B, H, T, _ = q.shape
    q = q.zero_()
    lens = torch.from_numpy(np.random.randint(1, T, size=B))
    mask = sequence_mask(lens, T).unsqueeze(1).unsqueeze(1)
    res, _ = attn(q, k, v, mask=mask)
    res = res.numpy()
    gold = v.numpy()
    for b in range(B):
        for h in range(H):
            for t in range(T):
                np.testing.assert_allclose(res[b, h, t, :], np.mean(gold[:, :, :lens[b], :], axis=2)[b, h, :], atol=1e-5)


def test_sdpa_values_sub_mask(qkv):
    attn_values_sub_mask(sdpa, qkv)


def test_dpa_values_sub_mask(qkv):
    attn_values_sub_mask(sdpa, qkv)


def attn_values_sub_mask(attn, qkv):
    q, k, v = qkv
    B, H, T, _ = q.shape
    q = q.zero_()
    mask = subsequent_mask(T)
    res, _ = attn(q, k, v, mask=mask)
    res = res.numpy()
    gold = v.numpy()
    for b in range(B):
        for h in range(H):
            for t in range(T):
                np.testing.assert_allclose(res[b, h, t, :], np.mean(gold[:, :, :t+1, :], axis=2)[b, h, :], atol=1e-5)

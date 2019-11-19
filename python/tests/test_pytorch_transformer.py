import copy
import pytest
import numpy as np
from mock import MagicMock
torch = pytest.importorskip('torch')
from baseline.pytorch.torchy import sequence_mask
from baseline.pytorch.transformer import subsequent_mask
from eight_mile.pytorch.layers import SeqScaledDotProductAttention, SeqDotProductAttention
# from baseline.pytorch.transformer import scaled_dot_product_attention as sdpa
# from baseline.pytorch.transformer import dot_product_attention as dpa



@pytest.fixture(autouse=True)
def no_grad():
    with torch.no_grad():
        yield


@pytest.fixture
def qkv():
    B, H, T, D = map(int, np.random.randint(5, 10, size=4))
    q = torch.rand(B, H, T, D)
    k = torch.rand(B, H, T, D)
    v = torch.rand(B, H, T, D)
    return q, k, v


def test_sdpa_values(qkv):
    sdpa = SeqScaledDotProductAttention(0.0)
    attn_values(sdpa, qkv)


def test_dpa_values(qkv):
    dpa = SeqDotProductAttention(0.0)
    attn_values(dpa, qkv)


def attn_values(attn, qkv):
    q, k, v = qkv
    B, H, T, _ = q.shape
    q = q.zero_()
    res = attn((q, k, v, None))
    res = res.numpy()
    gold = v.numpy()
    for b in range(B):
        for h in range(H):
            for t in range(T):
                np.testing.assert_allclose(res[b, h, t, :], np.mean(gold, axis=2)[b, h, :], rtol=1e-5)


def test_sdpa_values_seq_mask(qkv):
    sdpa = SeqScaledDotProductAttention(0.0)
    attn_values_seq_mask(sdpa, qkv)


def test_dpa_values_seq_mask(qkv):
    dpa = SeqDotProductAttention(0.0)
    attn_values_seq_mask(dpa, qkv)


def attn_values_seq_mask(attn, qkv):
    q, k, v = qkv
    B, H, T, _ = q.shape
    q = q.zero_()
    lens = torch.from_numpy(np.random.randint(1, T, size=B))
    mask = sequence_mask(lens, T).unsqueeze(1).unsqueeze(1)
    res = attn((q, k, v, mask))
    res = res.numpy()
    gold = v.numpy()
    for b in range(B):
        for h in range(H):
            for t in range(T):
                np.testing.assert_allclose(res[b, h, t, :], np.mean(gold[:, :, :lens[b], :], axis=2)[b, h, :], atol=1e-5)


def test_sdpa_values_sub_mask(qkv):
    sdpa = SeqScaledDotProductAttention(0.0)
    attn_values_sub_mask(sdpa, qkv)


def test_dpa_values_sub_mask(qkv):
    dpa = SeqDotProductAttention(0.0)
    attn_values_sub_mask(dpa, qkv)


def attn_values_sub_mask(attn, qkv):
    q, k, v = qkv
    B, H, T, _ = q.shape
    q = q.zero_()
    mask = subsequent_mask(T)
    res = attn((q, k, v, mask))
    res = res.numpy()
    gold = v.numpy()
    for b in range(B):
        for h in range(H):
            for t in range(T):
                np.testing.assert_allclose(res[b, h, t, :], np.mean(gold[:, :, :t+1, :], axis=2)[b, h, :], atol=1e-5)

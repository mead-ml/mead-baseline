import copy
import pytest
import numpy as np
from mock import MagicMock

torch = pytest.importorskip("torch")
from eight_mile.pytorch.layers import (
    SeqScaledDotProductAttention,
    SeqDotProductAttention,
    SeqDotProductRelativeAttention,
    SeqScaledDotProductRelativeAttention,
    sequence_mask,
    subsequent_mask,
)


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


@pytest.fixture
def ra_inputs():
    B, H, T, D = map(int, np.random.randint(5, 10, size=4))
    q = torch.rand(B, H, T, D)
    k = torch.rand(B, H, T, D)
    v = torch.rand(B, H, T, D)
    ek = torch.rand(T, T, D)
    ev = torch.rand(T, T, D)
    return q, k, v, ek, ev, None


def test_rel_attn_shapes(ra_inputs):
    ra = SeqScaledDotProductRelativeAttention()
    output = ra(ra_inputs)
    assert output.shape == ra_inputs[0].shape


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
                np.testing.assert_allclose(
                    res[b, h, t, :], np.mean(gold[:, :, : lens[b], :], axis=2)[b, h, :], atol=1e-5
                )


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
                np.testing.assert_allclose(res[b, h, t, :], np.mean(gold[:, :, : t + 1, :], axis=2)[b, h, :], atol=1e-5)

from collections import namedtuple
import numpy as np
import pytest
torch = pytest.importorskip("torch")

from eight_mile.pytorch.layers import (
    LuongDotProductAttention,
    ScaledDotProductAttention,
    LuongGeneralAttention,
    BahdanauAttention,
    sequence_mask,
)


qkvm = namedtuple("qkvm", "q k v m")
AttentionTypes = (LuongDotProductAttention, LuongGeneralAttention, ScaledDotProductAttention, BahdanauAttention)

@pytest.fixture
def generate_qkvm():
    B = np.random.randint(5, 10)
    T = np.random.randint(20, 30)
    H = np.random.randint(100, 200)

    B = 3
    T = 4
    H = 5

    q = torch.rand(B, H)
    k = torch.rand(B, T, H)
    v = torch.rand(B, T, H)

    lengths = torch.randint(1, T, size=(B,))
    lengths[torch.randint(0, B, size=(B//2,))] = T

    m = sequence_mask(lengths)

    ks = []
    vs = []
    ms = []
    for i, l in enumerate(lengths):
        k[i, l:] = 0
        v[i, l:] = 0
        ks.append(k[i, :l].unsqueeze(0))
        vs.append(v[i, :l].unsqueeze(0))
        ms.append(m[i, :l].unsqueeze(0))

    qs = [x.unsqueeze(0) for x in q]

    return qkvm(q, k, v, m), qkvm(qs, ks, vs, ms)


def join_attention_scores(scores, max_len):
    full = torch.zeros((len(scores), max_len), dtype=scores[0].dtype)
    for i, score in enumerate(scores):
        full[i, :score.shape[1]] = score[0]
    return full


def generate_scores(masks):
    scores = []
    for mask in masks:
        score = torch.randn(*mask.shape)
        score = score / torch.sum(score)
        scores.append(score)
    return join_attention_scores(scores, max(x.shape[-1] for x in scores)), scores


def test_attention_batch_stability(generate_qkvm):
    batched, single = generate_qkvm

    def test_full_attention(batched, single, attn):
        batched_result = attn(*batched)
        single_results = torch.cat([attn(qi, ki, vi, mi) for qi, ki, vi, mi in zip(*single)], dim=0)
        np.testing.assert_allclose(batched_result.detach().numpy(), single_results.detach().numpy(), atol=1e-6)

    for Attn in AttentionTypes:
        attn = Attn(batched.q.shape[-1])
        print(attn.__class__.__name__)

        test_full_attention(batched, single, attn)


def test_attention_scores_batch_stability(generate_qkvm):
    batched, single = generate_qkvm

    def test_attention_scores(batched, single, attn):
        batched_scores = attn._attention(batched.q, batched.k, batched.m)
        single_scores = [attn._attention(qi, ki, mi) for qi, ki, mi in zip(single.q, single.k, single.m)]
        single_scores = join_attention_scores(single_scores, max(x.shape[-1] for x in single_scores))
        np.testing.assert_allclose(batched_scores.detach().numpy(), single_scores.detach().numpy(), atol=1e-6)

    for Attn in AttentionTypes:
        attn = Attn(batched.q.shape[-1])
        print(attn.__class__.__name__)

        test_attention_scores(batched, single, attn)


def test_attention_combination_stability(generate_qkvm):
    batched, single = generate_qkvm
    batched_scores, single_scores = generate_scores(single.m)

    def test_attention_combination(batched_scores, batched, single_scores, single, attn):
        batched_combo = attn._update(batched_scores, batched.q, batched.v)
        single_combo = torch.cat([attn._update(score, qi, vi) for score, qi, vi in zip(single_scores, single.q, single.v)], dim=0)
        np.testing.assert_allclose(batched_combo.detach().numpy(), single_combo.detach().numpy(), atol=1e-6)

    for Attn in AttentionTypes:
        attn = Attn(batched.q.shape[-1])
        print(attn.__class__.__name__)

        test_attention_combination(batched_scores, batched, single_scores, single, attn)

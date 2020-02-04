import os
import math
import json
import pytest
import numpy as np
from mock import patch, MagicMock

torch = pytest.importorskip("torch")
from eight_mile.utils import Offsets
from eight_mile.pytorch.layers import (
    CRF,
    Viterbi,
    ViterbiLogSoftmaxNorm,
    transition_mask,
    script_viterbi,
    ViterbiBatchSize1,
    vec_log_sum_exp,
)


def explicit_log_sum_exp(xs):
    """Log Sum Exp on a dict of values."""
    max_x = max(xs.values())
    total = 0
    for x in xs.values():
        total += math.exp(x - max_x)
    return max_x + math.log(total)


def explicit_score_gold(emiss, trans, golds, start, end):
    score = 0
    for e, g in zip(emiss, golds):
        score += e[g]
    for i in range(len(golds)):
        from_ = start if i == 0 else golds[i - 1]
        to = golds[i]
        score += trans[(from_, to)]
    score += trans[(golds[-1], end)]
    return score


def explicit_forward(emiss, trans, start, end):
    """Best path through a lattice on the log semiring with explicit looping."""
    trellis = dict.fromkeys(emiss[0].keys(), -1e4)
    trellis[start] = 0

    for e in emiss:
        new_trellis = {}
        for next_state in trellis:
            score = {}
            for prev_state in trellis:
                score[prev_state] = trellis[prev_state] + e[next_state] + trans[(prev_state, next_state)]
            new_trellis[next_state] = explicit_log_sum_exp(score)
        trellis = new_trellis
    for state in trellis:
        trellis[state] += trans[(state, end)]
    return explicit_log_sum_exp(trellis)


def explicit_nll(emiss, trans, golds, start, end):
    f = explicit_forward(emiss, trans, start, end)
    g = explicit_score_gold(emiss, trans, golds, start, end)
    return f - g


def explicit_viterbi(emiss, trans, start, end):
    """Best path through a lattice on the viterbi semiring with explicit looping."""
    backpointers = []
    trellis = dict.fromkeys(emiss[0].keys(), -1e4)
    trellis[start] = 0

    for e in emiss:
        new_trellis = {}
        backpointer = {}
        for next_state in trellis:
            score = {}
            for prev_state in trellis:
                score[prev_state] = trellis[prev_state] + e[next_state] + trans[(prev_state, next_state)]
            new_trellis[next_state] = max(score.values())
            backpointer[next_state] = max(score, key=lambda x: score[x])  # argmax
        trellis = new_trellis
        backpointers.append(backpointer)
    for state in trellis:
        trellis[state] += trans[(state, end)]
    score = max(trellis.values())
    state = max(trellis, key=lambda x: trellis[x])
    states = [state]
    for t in reversed(range(0, len(emiss))):
        states.append(backpointers[t][states[-1]])
    return list(reversed(states[:-1])), score


def build_trans(t):
    """Convert the transition tensor to a dict.

    :param t: `torch.FloatTensor` [H, H]: transition scores in the
        form [to, from]

    :returns: `dict` transition scores in the form [from, to]
    """
    trans = {}
    for i in range(t.size(0)):
        for j in range(t.size(1)):
            trans[(i, j)] = t[j, i].item()
    return trans


def build_emission(emission):
    """Convert the emission scores into a list of dicts

    :param emission: `torch.FloatTensor` [T, H]: emission scores

    :returns: `List[dict]`
    """
    es = []
    for emiss in emission:
        e_ = {}
        for i in range(emiss.size(0)):
            e_[i] = emiss[i].item()
        es.append(e_)
    return es


@pytest.fixture
def generate_batch():
    """Generate a batch of data.

    Creates lengths such that at least one half of the batch is the maximum
    length.

    :returns: unary [B, T, H], tags [T, B], lengths [B]
    """
    B = np.random.randint(5, 11)
    T = np.random.randint(15, 21)
    H = np.random.randint(22, 41)
    scores = torch.rand(B, T, H)
    tags = torch.randint(1, H, (B, T))
    lengths = torch.randint(1, T, (B,))
    lengths[torch.randint(0, B, (B // 2,))] = T
    for s, l in zip(scores, lengths):
        s[l:] = 0
    return scores.transpose(0, 1), tags.transpose(0, 1), lengths


@pytest.fixture
def generate_examples_and_batch():
    """A good test for these systems are do they produce the same results for a
    batch of data as when you feed the example in one by one.

    This function generates two single examples and then batches them together.
    """
    T = np.random.randint(15, 21)
    H = np.random.randint(22, 41)
    diff = np.random.randint(1, T // 2)

    item1 = torch.rand(T, 1, H)
    tags1 = torch.randint(1, H, (T, 1))
    lengths1 = torch.tensor([T])

    item2 = torch.rand(T - diff, 1, H)
    tags2 = torch.randint(1, H, (T - diff, 1))
    lengths2 = torch.tensor([T - diff])

    packed_input = torch.zeros(T, 2, H)
    packed_tags = torch.zeros(T, 2, dtype=torch.long)
    packed_input[:, 0, :] = item1.squeeze(1)
    packed_input[: T - diff, 1, :] = item2.squeeze(1)
    packed_tags[:, 0] = tags1.squeeze(1)
    packed_tags[: T - diff, 1] = tags2.squeeze(1)
    lengths = torch.cat([lengths1, lengths2], dim=0)
    return item1, tags1, lengths1, item2, tags2, lengths2, packed_input, packed_tags, lengths


def test_score_sentence(generate_batch):
    unary, tags, lengths = generate_batch
    h = unary.size(2)
    crf = CRF(h, batch_first=False)
    trans = torch.rand(h, h)
    crf.transitions_p.data = trans.unsqueeze(0)
    sentence_score = crf.score_sentence(unary, tags, lengths)

    new_trans = build_trans(trans)
    unary = unary.transpose(0, 1)
    tags = tags.transpose(0, 1)
    scores = []
    for u, t, l in zip(unary, tags, lengths):
        emiss = build_emission(u[:l])
        golds = t[:l].tolist()
        scores.append(explicit_score_gold(emiss, new_trans, golds, Offsets.GO, Offsets.EOS))
    gold_scores = np.array(scores)
    np.testing.assert_allclose(sentence_score.detach().numpy(), gold_scores, rtol=1e-6)


def test_score_sentence_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    h = i1.size(2)
    crf = CRF(h, batch_first=False)
    crf.transitions_p.data = torch.rand(1, h, h)
    score1 = crf.score_sentence(i1, t1, l1)
    score2 = crf.score_sentence(i2, t2, l2)
    one_x_one = torch.cat([score1, score2], dim=0)
    batched = crf.score_sentence(i, t, l)
    np.testing.assert_allclose(one_x_one.detach().numpy(), batched.detach().numpy())


def test_score_sentence_shape(generate_batch):
    unary, tags, lengths = generate_batch
    h = unary.size(2)
    crf = CRF(h, batch_first=False)
    score = crf.score_sentence(unary, tags, lengths)
    assert score.shape == torch.Size([unary.size(1)])


# def test_neg_log_loss(generate_batch):
#    unary, tags, lengths = generate_batch
#    h = unary.size(2)
#    crf = CRF(h, batch_first=False)
#    trans = torch.rand(h, h)
#    crf.transitions_p.data = trans.unsqueeze(0)
#    nll = crf.neg_log_loss(unary, tags, lengths)

#    new_trans = build_trans(trans)
#    unary = unary.transpose(0, 1)
#    tags = tags.transpose(0, 1)
#    scores = []
#    for u, t, l in zip(unary, tags, lengths):
#        emiss = build_emission(u[:l])
#        golds = t[:l].tolist()
#        scores.append(explicit_nll(emiss, new_trans, golds, Offsets.GO, Offsets.EOS))
#    gold_scores = np.array(scores)
#    np.testing.assert_allclose(nll.detach().numpy(), gold_scores, rtol=1e-6)


# def test_neg_log_loss_batch_stable(generate_examples_and_batch):
#    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
#    h = i1.size(2)
#    crf = CRF(h, batch_first=False)
#    crf.transitions_p.data = torch.rand(1, h, h)
#    nll1 = crf.neg_log_loss(i1, t1, l1)
#    nll2 = crf.neg_log_loss(i2, t2, l2)
#    one_x_one = torch.cat([nll1, nll2], dim=0)
#    batched = crf.neg_log_loss(i, t, l)
#    np.testing.assert_allclose(one_x_one.detach().numpy(), batched.detach().numpy())


def test_forward(generate_batch):
    unary, _, lengths = generate_batch
    h = unary.size(2)
    crf = CRF(h, batch_first=False)
    trans = torch.rand(h, h)
    crf.transitions_p.data = trans.unsqueeze(0)
    forward = crf.forward((unary, lengths))

    new_trans = build_trans(trans)
    unary = unary.transpose(0, 1)
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        scores.append(explicit_forward(emiss, new_trans, Offsets.GO, Offsets.EOS))
    gold_scores = np.array(scores)
    np.testing.assert_allclose(forward.detach().numpy(), gold_scores, rtol=1e-6)


def test_forward_batch_stable(generate_examples_and_batch):
    i1, _, l1, i2, _, l2, i, _, l = generate_examples_and_batch
    h = i1.size(2)
    crf = CRF(h, batch_first=False)
    crf.transitions_p.data = torch.rand(1, h, h)
    fw1 = crf.forward((i1, l1))
    fw2 = crf.forward((i2, l2))
    one_x_one = torch.cat([fw1, fw2], dim=0)
    batched = crf.forward((i, l))
    np.testing.assert_allclose(one_x_one.detach().numpy(), batched.detach().numpy())


def test_forward_shape(generate_batch):
    unary, _, lengths = generate_batch
    h = unary.size(2)
    crf = CRF(h, batch_first=False)
    fwd = crf.forward((unary, lengths))
    assert fwd.shape == torch.Size([unary.size(1)])


# def test_decode_batch_stable(generate_examples_and_batch):
#    i1, _, l1, i2, _, l2, i, _, l = generate_examples_and_batch
#    h = i1.size(2)
#    crf = CRF(h, batch_first=False)
#    crf.transitions_p.data = torch.rand(1, h, h)
#    p1 = crf.decode(i1, l1)
#    p2 = crf.decode(i2, l2)
#    pad = torch.zeros(p1.size(0) - p2.size(0), 1, dtype=torch.long)
#    one_x_one_p = torch.cat([p1, torch.cat([p2, pad], dim=0)], dim=1)
#    one_x_one_s = torch.cat([s1, s2], dim=0)
#    batched_p crf.decode(i, l)
#    #np.testing.assert_allclose(one_x_one_s.detach().numpy(), batched_s.detach().numpy())
#    for p1, p2 in zip(one_x_one_p, batched_p):
#        np.testing.assert_allclose(p1.detach().numpy(), p2.detach().numpy())


def test_decode_shape_crf(generate_batch):
    unary, _, lengths = generate_batch
    h = unary.size(2)
    crf = CRF(h, batch_first=False)
    paths, scores = crf.decode(unary, lengths)
    assert scores.shape == torch.Size([unary.size(1)])
    assert paths.shape == torch.Size([unary.size(0), unary.size(1)])


def test_mask_is_applied():
    h = np.random.randint(22, 41)
    loc = np.random.randint(h)
    constraint = torch.zeros(h, h, dtype=torch.uint8)
    constraint[Offsets.GO, loc] = 1
    crf = CRF(h, constraint_mask=constraint)
    t = crf.transitions.detach().numpy()
    assert t[0, Offsets.GO, loc] == -1e4


def test_mask_not_applied():
    h = np.random.randint(22, 41)
    crf = CRF(h)
    t = crf.transitions.detach().numpy()
    assert t[0, Offsets.GO, np.random.randint(h)] != -1e4


def test_mask_flips():
    h = np.random.randint(22, 41)
    mask = (np.random.rand(h, h) < 0.5).astype(np.uint8)
    with patch("eight_mile.pytorch.layers.transition_mask_np") as mask_mock:
        mask_mock.return_value = mask
        pyt_mask = transition_mask(None, None, None, None, None)
    mask2 = pyt_mask.numpy()
    assert (mask & mask2).sum() == 0
    assert (mask | mask2).sum() == h * h


def test_mask_same_after_update(generate_batch):
    from torch.optim import SGD

    unary, tags, lengths = generate_batch
    h = unary.size(2)
    constraint = torch.rand(h, h) < 0.5
    crf = CRF(h, constraint_mask=constraint, batch_first=False)
    opt = SGD(crf.parameters(), lr=10)
    m1 = crf.constraint_mask.numpy()
    t1 = crf.transitions_p.detach().clone().numpy()
    l = crf.neg_log_loss(unary, tags, lengths)
    l = torch.mean(l)
    l.backward()
    opt.step()
    m2 = crf.constraint_mask.numpy()
    t2 = crf.transitions_p.detach().numpy()
    np.testing.assert_allclose(m1, m2)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(t1, t2)


def test_viterbi(generate_batch):
    unary, _, lengths = generate_batch
    h = unary.size(2)
    trans = torch.rand(h, h)
    pyt_path, pyt_scores = Viterbi(Offsets.GO, Offsets.EOS)(unary, trans.unsqueeze(0), lengths)

    new_trans = build_trans(trans)
    unary = unary.transpose(0, 1)
    paths = []
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        p, s = explicit_viterbi(emiss, new_trans, Offsets.GO, Offsets.EOS)
        scores.append(s)
        paths.append(p)
    gold_scores = np.array(scores)
    np.testing.assert_allclose(pyt_scores.detach().numpy(), gold_scores, rtol=1e-6)
    pyt_path = pyt_path.transpose(0, 1)
    for pp, l, p in zip(pyt_path, lengths, paths):
        assert pp[:l].tolist() == p


def test_viterbi_script(generate_batch):
    unary, _, lengths = generate_batch
    h = unary.size(2)
    trans = torch.rand(h, h)

    # pyt_path, pyt_scores = ViterbiBatchSize1(Offsets.GO, Offsets.EOS)(unary, trans.unsqueeze(0), lengths)

    new_trans = build_trans(trans)
    batch_first_unary = unary.transpose(0, 1)
    for u, l in zip(batch_first_unary, lengths):
        emiss = build_emission(u[:l])
        p, s = explicit_viterbi(emiss, new_trans, Offsets.GO, Offsets.EOS)
        ps, ss = script_viterbi(u[:l], trans, Offsets.GO, Offsets.EOS)

        np.testing.assert_allclose(ps.numpy().tolist(), p, rtol=1e-6)
        np.testing.assert_allclose(ss.item(), s, rtol=1e-6)


def test_viterbi_batch_stable(generate_examples_and_batch):
    i1, _, l1, i2, _, l2, i, _, l = generate_examples_and_batch
    h = i1.size(2)
    trans = torch.rand(1, h, h)
    p1, s1 = Viterbi(Offsets.GO, Offsets.EOS)(i1, trans, l1)
    p2, s2 = Viterbi(Offsets.GO, Offsets.EOS)(i2, trans, l2)
    pad = torch.zeros(p1.size(0) - p2.size(0), 1, dtype=torch.long)
    one_x_one_p = torch.cat([p1, torch.cat([p2, pad], dim=0)], dim=1)
    one_x_one_s = torch.cat([s1, s2], dim=0)
    batched_p, batched_s = Viterbi(Offsets.GO, Offsets.EOS)(i, trans, l)
    np.testing.assert_allclose(one_x_one_s.detach().numpy(), batched_s.detach().numpy())
    np.testing.assert_allclose(one_x_one_p.detach().numpy(), batched_p.detach().numpy())


def test_viterbi_degenerates_to_argmax(generate_batch):
    scores, _, l = generate_batch
    h = scores.size(2)
    # Then transitions are all zeros then it just greedily selects the best
    # state at that given emission. This is the same as doing argmax.
    trans = torch.zeros((1, h, h))
    viterbi = Viterbi(Offsets.GO, Offsets.EOS)
    p, s = viterbi(scores, trans, l)
    s_gold, p_gold = torch.max(scores, 2)
    # Mask out the argmax results from past the lengths
    for i, sl in enumerate(l):
        s_gold[sl:, i] = 0
        p_gold[sl:, i] = 0
    s_gold = torch.sum(s_gold, 0)
    np.testing.assert_allclose(p.detach().numpy(), p_gold.detach().numpy())
    np.testing.assert_allclose(s.detach().numpy(), s_gold.detach().numpy())


def test_decode_shape(generate_batch):
    unary, _, lengths = generate_batch
    h = unary.size(2)
    trans = torch.rand(1, h, h)
    viterbi = Viterbi(Offsets.GO, Offsets.EOS)
    paths, scores = viterbi(unary, trans, lengths)
    assert scores.shape == torch.Size([unary.size(1)])
    assert paths.shape == torch.Size([unary.size(0), unary.size(1)])


def test_vec_log_sum_exp():
    vec = torch.rand(1, np.random.randint(5, 31))
    ours = vec_log_sum_exp(vec, 1).squeeze()
    xs = {}
    for i in range(vec.size(1)):
        xs[i] = vec[0, i].item()
    gold = explicit_log_sum_exp(xs)
    np.testing.assert_allclose(ours, gold, rtol=1e-6)


def test_vec_log_sum_exp_zeros():
    l = np.random.randint(1, 21)
    in_ = torch.zeros(1, l)
    lse = vec_log_sum_exp(in_, 1).squeeze()
    np.testing.assert_allclose(lse.detach().numpy(), math.log(l))


def test_vec_log_sum_exp_ones():
    l = np.random.randint(1, 21)
    in_ = torch.ones(1, l)
    lse = vec_log_sum_exp(in_, 1).squeeze()
    np.testing.assert_allclose(lse.detach().numpy(), math.log(l * math.e))


def test_vec_log_sum_exp_shape():
    dim = torch.randint(0, 3, (1,)).item()
    shape = torch.randint(1, 21, (3,))
    in_ = torch.rand(*shape)
    out = vec_log_sum_exp(in_, dim)
    shape[dim] = 1
    for i in range(len(shape)):
        assert out.size(i) == shape[i]


def test_vec_log_sum_exp_batch_stable():
    h = np.random.randint(22, 41)
    i1 = torch.rand(1, h, h)
    i2 = torch.rand(1, h, h)
    i = torch.cat([i1, i2], dim=0)
    lse1 = vec_log_sum_exp(i1, 2)
    lse2 = vec_log_sum_exp(i2, 2)
    one_x_one = torch.cat([lse1, lse2], dim=0)
    lse = vec_log_sum_exp(i, 2)
    np.testing.assert_allclose(one_x_one.numpy(), lse.numpy())

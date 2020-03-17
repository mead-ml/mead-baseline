import os
import math
import json
from operator import itemgetter
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
from tagger_decode_utils import (
    explicit_log_sum_exp,
    explicit_score_gold,
    explicit_forward,
    explicit_backward,
    explicit_posterior,
    explicit_posterior_decode,
    explicit_trellises_to_dense,
    explicit_trellis_to_dense,
    explicit_nll,
    explicit_viterbi,
    build_trans,
    build_emission,
    generate_batch as make_batch,
    generate_examples_and_batch as make_examples_and_batch,
)


@pytest.fixture
def generate_batch():
    """Generate a batch of data.

    Creates lengths such that at least one half of the batch is the maximum
    length.

    :returns: unary [B, T, H], tags [T, B], lengths [B]
    """
    scores, tags, lengths = make_batch()
    scores = torch.from_numpy(scores)
    tags = torch.from_numpy(tags)
    lengths = torch.from_numpy(lengths)
    return scores.transpose(0, 1), tags.transpose(0, 1), lengths


@pytest.fixture
def generate_examples_and_batch():
    """A good test for these systems are do they produce the same results for a
    batch of data as when you feed the example in one by one.

    This function generates two single examples and then batches them together.
    """
    i1, t1, l1, i2, t2, l2, items, ts, lengths = map(torch.from_numpy, make_examples_and_batch())
    i1 = i1.transpose(0, 1)
    t1 = t1.transpose(0, 1)
    i2 = i2.transpose(0, 1)
    t2 = t2.transpose(0, 1)
    items = items.transpose(0, 1)
    ts = ts.transpose(0, 1)
    return i1, t1, l1, i2, t2, l2, items, ts, lengths


def make_crf(unary):
    h = unary.size(2)
    crf = CRF(h, batch_first=False)
    trans = torch.rand(h, h)
    crf.transitions_p.data = trans.unsqueeze(0)
    return crf, trans


def test_score_sentence(generate_batch):
    unary, tags, lengths = generate_batch
    crf, trans = make_crf(unary)

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
    crf, _ = make_crf(i)

    score1 = crf.score_sentence(i1, t1, l1)
    score2 = crf.score_sentence(i2, t2, l2)
    one_x_one = torch.cat([score1, score2], dim=0)
    batched = crf.score_sentence(i, t, l)
    np.testing.assert_allclose(one_x_one.detach().numpy(), batched.detach().numpy())


def test_score_sentence_shape(generate_batch):
    unary, tags, lengths = generate_batch
    crf, _ = make_crf(unary)

    score = crf.score_sentence(unary, tags, lengths)
    assert score.shape == torch.Size([unary.size(1)])


def test_neg_log_loss(generate_batch):
   unary, tags, lengths = generate_batch
   crf, trans = make_crf(unary)

   nll = crf.neg_log_loss(unary, tags, lengths)

   new_trans = build_trans(trans)
   unary = unary.transpose(0, 1)
   tags = tags.transpose(0, 1)
   scores = []
   for u, t, l in zip(unary, tags, lengths):
       emiss = build_emission(u[:l])
       golds = t[:l].tolist()
       scores.append(explicit_nll(emiss, new_trans, golds, Offsets.GO, Offsets.EOS))
   gold_scores = np.mean(np.array(scores))
   np.testing.assert_allclose(nll.detach().numpy(), gold_scores, rtol=1e-6)


def test_neg_log_loss_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    nll1 = crf.neg_log_loss(i1, t1, l1)
    nll2 = crf.neg_log_loss(i2, t2, l2)
    one_x_one = (nll1 + nll2) / 2
    batched = crf.neg_log_loss(i, t, l)
    np.testing.assert_allclose(one_x_one.detach().numpy(), batched.detach().numpy())


def test_forward(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    forward = crf.partition(unary, lengths)

    new_trans = build_trans(trans)
    unary = unary.transpose(0, 1)
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        scores.append(explicit_forward(emiss, new_trans, Offsets.GO, Offsets.EOS)[0])
    gold_scores = np.array(scores)
    np.testing.assert_allclose(forward.detach().numpy(), gold_scores, rtol=1e-6)


def test_forward_batch_stable(generate_examples_and_batch):
    i1, _, l1, i2, _, l2, i, _, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    fw1 = crf.partition(i1, l1)
    fw2 = crf.partition(i2, l2)
    one_x_one = torch.cat([fw1, fw2], dim=0)
    batched = crf.partition(i, l)
    np.testing.assert_allclose(one_x_one.detach().numpy(), batched.detach().numpy())


def test_forward_shape(generate_batch):
    unary, _, lengths = generate_batch
    crf, _ = make_crf(unary)
    fwd = crf.partition(unary, lengths)
    assert fwd.shape == torch.Size([unary.size(1)])


def test_forward_over_time(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    forward = crf.partition_over_time(unary, lengths).transpose(0, 1)

    new_trans = build_trans(trans)
    unary = unary.transpose(0, 1)
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        scores.append(explicit_forward(emiss, new_trans, Offsets.GO, Offsets.EOS)[1])
    for fwd, gold, l in zip(forward, scores, lengths):
        fwd = fwd[:l, :]
        np.testing.assert_allclose(fwd.detach().numpy(), explicit_trellises_to_dense(gold), rtol=1e-6)


def test_forward_over_time_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    f1 = crf.partition_over_time(i1, l1).transpose(0, 1)
    f2 = crf.partition_over_time(i2, l2).transpose(0, 1)

    batched = crf.partition_over_time(i, l)

    one_x_one = torch.zeros((2, f1.shape[1], f1.shape[2]))
    one_x_one[0, :f1.shape[1]] = f1.squeeze(0)
    one_x_one[1, :f2.shape[1]] = f2.squeeze(0)
    one_x_one = one_x_one.transpose(0, 1)

    np.testing.assert_allclose(one_x_one.detach().numpy(), batched.detach().numpy(), rtol=1e-6)


def test_backward_over_time(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    backward = crf.partition_backward_over_time(unary, lengths).transpose(0, 1)

    new_trans = build_trans(trans)
    unary = unary.transpose(0, 1)
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        scores.append(explicit_backward(emiss, new_trans, Offsets.GO, Offsets.EOS)[1])
    for bwd, gold, l in zip(backward, scores, lengths):
        bwd = bwd[:l, :]
        np.testing.assert_allclose(bwd.detach().numpy(), explicit_trellises_to_dense(gold), rtol=1e-6)


def test_backward_over_time_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    f1 = crf.partition_backward_over_time(i1, l1).transpose(0, 1)
    f2 = crf.partition_backward_over_time(i2, l2).transpose(0, 1)

    batched = crf.partition_backward_over_time(i, l)

    one_x_one = torch.zeros((2, f1.shape[1], f1.shape[2]))
    one_x_one[0, :f1.shape[1]] = f1.squeeze(0)
    one_x_one[1, :f2.shape[1]] = f2.squeeze(0)
    one_x_one = one_x_one.transpose(0, 1)

    np.testing.assert_allclose(one_x_one.detach().numpy(), batched.detach().numpy(), rtol=1e-6)


def test_posterior(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    posterior = crf.posterior(unary, lengths).transpose(0, 1)

    new_trans = build_trans(trans)
    unary = unary.transpose(0, 1)
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        scores.append(explicit_posterior(emiss, new_trans, Offsets.GO, Offsets.EOS))
    for post, gold, l in zip(posterior, scores, lengths):
        post = post[:l, :]
        np.testing.assert_allclose(post.detach().numpy(), explicit_trellises_to_dense(gold), rtol=1e-6)


def test_posterior_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    p1 = crf.posterior(i1, l1).transpose(0, 1)
    p2 = crf.posterior(i2, l2).transpose(0, 1)

    batched = crf.posterior(i, l)

    one_x_one = torch.zeros((2, p1.shape[1], p1.shape[2]))
    one_x_one[0, :p1.shape[1]] = p1.squeeze(0)
    one_x_one[1, :p2.shape[1]] = p2.squeeze(0)
    one_x_one = one_x_one.transpose(0, 1)

    np.testing.assert_allclose(batched.detach().numpy(), one_x_one.detach().numpy(), rtol=1e-6)


def test_posterior_decode(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    paths, scores = crf.posterior_decode(unary, lengths)
    paths = paths.transpose(0, 1)

    new_trans = build_trans(trans)
    unary = unary.transpose(0, 1)
    gold_paths = []
    gold_scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        p, s = explicit_posterior_decode(emiss, new_trans, Offsets.GO, Offsets.EOS)
        gold_paths.append(p)
        gold_scores.append(s)
    for p, g, l in zip(paths, gold_paths, lengths):
        p = p[:l]
        np.testing.assert_allclose(p.detach().numpy(), np.array(g), rtol=1e-6)
    np.testing.assert_allclose(scores.detach().numpy(), np.array(gold_scores), rtol=1e-6)


def test_posterior_decode_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    p1, s1 = crf.posterior_decode(i1, l1)
    p2, s2 = crf.posterior_decode(i2, l2)

    p1 = p1.transpose(0, 1)
    p2 = p2.transpose(0, 1)

    batched_p, batched_s = crf.posterior_decode(i, l)

    one_x_one_s = torch.cat([s1, s2], dim=0)

    one_x_one_p = torch.zeros((2, p1.shape[1]), dtype=p1.dtype)
    one_x_one_p[0, :p1.shape[1]] = p1.squeeze(0)
    one_x_one_p[1, :p2.shape[1]] = p2.squeeze(0)
    one_x_one_p = one_x_one_p.transpose(0, 1)

    np.testing.assert_allclose(batched_s.detach().numpy(), one_x_one_s.detach().numpy(), rtol=1e-6)
    np.testing.assert_allclose(batched_p.detach().numpy(), one_x_one_p.detach().numpy(), rtol=1e-6)

def test_decode_batch_stable(generate_examples_and_batch):
    i1, _, l1, i2, _, l2, i, _, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    p1, s1 = crf.decode(i1, l1)
    p2, s2 = crf.decode(i2, l2)
    pad = torch.zeros(p1.size(0) - p2.size(0), 1, dtype=torch.long)
    one_x_one_p = torch.cat([p1, torch.cat([p2, pad], dim=0)], dim=1)
    one_x_one_s = torch.cat([s1, s2], dim=0)
    batched_p, batched_s =  crf.decode(i, l)
    np.testing.assert_allclose(one_x_one_s.detach().numpy(), batched_s.detach().numpy())
    for p1, p2 in zip(one_x_one_p, batched_p):
        np.testing.assert_allclose(p1.detach().numpy(), p2.detach().numpy())


def test_decode_shape_crf(generate_batch):
    unary, _, lengths = generate_batch
    crf, _ = make_crf(unary)

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


def test_viterbi_score_equals_sentence_score(generate_batch):
    """Test that the scores from viterbi decoding are the same scores that you get when looking up those returned paths."""
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    p, viterbi_scores = Viterbi(Offsets.GO, Offsets.EOS)(unary, crf.transitions, lengths)
    gold_scores = crf.score_sentence(unary, p, lengths)
    np.testing.assert_allclose(viterbi_scores.detach().numpy(), gold_scores.detach().numpy(), rtol=1e-6)


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

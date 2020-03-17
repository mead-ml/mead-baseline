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
    ConstrainedGreedyTaggerDecoder,
    Viterbi,
    ViterbiLogSoftmaxNorm,
    transition_mask,
    script_viterbi,
    ViterbiBatchSize1,
    vec_log_sum_exp,
)
from tagger_decode_utils import (
    explicit_log_sum_exp,
    explicit_sum,
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
    unary, tags, lengths = map(torch.from_numpy, make_batch())
    return unary, tags, lengths


@pytest.fixture
def generate_examples_and_batch():
    i1, t1, l1, i2, t2, l2, items, tags, lengths = map(torch.from_numpy, make_examples_and_batch())
    return i1, t1, l1, i2, t2, l2, items, tags, lengths


def make_constrained(input):
    size = input.shape[2]
    trans = torch.rand(size, size).bernoulli().to(torch.bool)
    cd = ConstrainedGreedyTaggerDecoder(size, trans.unsqueeze(0), batch_first=True)
    return cd, cd.transitions.squeeze(0)

def test_score_sentence(generate_batch):
    unary, tags, lengths = generate_batch
    cd, trans = make_constrained(unary)

    sentence_score = cd.score_sentence(unary, tags, lengths)

    new_trans = build_trans(trans)
    scores = []
    for u, t, l in zip(unary, tags, lengths):
        emiss = build_emission(u[:l])
        golds = t[:l].tolist()
        scores.append(explicit_score_gold(emiss, new_trans, golds, Offsets.GO, Offsets.EOS))
    gold_scores = np.array(scores)
    np.testing.assert_allclose(sentence_score.detach().numpy(), gold_scores, rtol=1e-6)


def test_score_sentence_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    cd, _ = make_constrained(i)

    score1 = cd.score_sentence(i1, t1, l1)
    score2 = cd.score_sentence(i2, t2, l2)
    one_x_one = torch.cat([score1, score2], dim=0)
    batched = cd.score_sentence(i, t, l)
    np.testing.assert_allclose(one_x_one.detach().numpy(), batched.detach().numpy())


def test_score_sentence_shape(generate_batch):
    unary, tags, lengths = generate_batch
    cd, _ = make_constrained(unary)

    score = cd.score_sentence(unary, tags, lengths)
    assert score.shape == torch.Size([unary.size(0)])


def test_partition(generate_batch):
    unary, _, lengths = generate_batch
    cd, trans = make_constrained(unary)
    partition = cd.partition(unary, lengths)

    new_trans = build_trans(trans)
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        scores.append(explicit_forward(emiss, new_trans, Offsets.GO, Offsets.EOS, explicit_sum)[0])
    gold_scores = np.array(scores)
    np.testing.assert_allclose(partition.detach().numpy(), gold_scores, rtol=1e-5, atol=1e-5)


def test_partition_batch_stable(generate_examples_and_batch):
    i1, _, l1, i2, _, l2, i, _, l = generate_examples_and_batch
    cd, trans = make_constrained(i)
    part1 = cd.partition(i1, l1)
    part2 = cd.partition(i2, l2)
    one_x_one = torch.cat([part1, part2], dim=0)
    batched = cd.partition(i, l)
    np.testing.assert_allclose(one_x_one.detach().numpy(), batched.detach().numpy())


def test_patition_shape(generate_batch):
    unary, _, lengths = generate_batch
    cd, trans = make_constrained(unary)
    partition = cd.partition(unary, lengths)
    assert partition.shape == torch.Size([unary.size(0)])


def test_partition_over_time(generate_batch):
    unary, _, lengths = generate_batch
    cd, trans = make_constrained(unary)
    forward = cd.partition_over_time(unary, lengths)
    new_trans = build_trans(trans)
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        scores.append(explicit_forward(emiss, new_trans, Offsets.GO, Offsets.EOS, explicit_sum)[1])
    for fwd, gold, l in zip(forward, scores, lengths):
        fwd = fwd[:l, :]
        np.testing.assert_allclose(fwd.detach().numpy(), explicit_trellises_to_dense(gold), rtol=1e-5, atol=1e-5)


def test_forward_over_time_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    cd, _ = make_constrained(i)

    f1 = cd.partition_over_time(i1, l1)
    f2 = cd.partition_over_time(i2, l2)

    batched = cd.partition_over_time(i, l)

    one_x_one = torch.zeros((2, f1.shape[1], f1.shape[2]))
    one_x_one[0, :f1.shape[1]] = f1.squeeze(0)
    one_x_one[1, :f2.shape[1]] = f2.squeeze(0)
    one_x_one = one_x_one

    np.testing.assert_allclose(one_x_one.detach().numpy(), batched.detach().numpy(), rtol=1e-6)


def test_backward_over_time(generate_batch):
    unary, _, lengths = generate_batch
    cd, trans = make_constrained(unary)

    backward = cd.partition_backward_over_time(unary, lengths)

    new_trans = build_trans(trans)
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        scores.append(explicit_backward(emiss, new_trans, Offsets.GO, Offsets.EOS, explicit_sum)[1])
    for bwd, gold, l in zip(backward, scores, lengths):
        bwd = bwd[:l, :]
        np.testing.assert_allclose(bwd.detach().numpy(), explicit_trellises_to_dense(gold), rtol=1e-5, atol=1e-6)


def test_backward_over_time_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    cd, _ = make_constrained(i)

    part1 = cd.partition_backward_over_time(i1, l1)
    part2 = cd.partition_backward_over_time(i2, l2)

    batched = cd.partition_backward_over_time(i, l)

    one_x_one = torch.zeros((2, part1.shape[1], part1.shape[2]))
    one_x_one[0, :part1.shape[1]] = part1.squeeze(0)
    one_x_one[1, :part2.shape[1]] = part2.squeeze(0)

    np.testing.assert_allclose(one_x_one.detach().numpy(), batched.detach().numpy(), rtol=1e-6)


def test_posterior(generate_batch):
    unary, _, lengths = generate_batch
    cd, trans = make_constrained(unary)

    posterior = cd.posterior(unary, lengths)

    new_trans = build_trans(trans)
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        scores.append(explicit_posterior(emiss, new_trans, Offsets.GO, Offsets.EOS, explicit_sum))
    for post, gold, l in zip(posterior, scores, lengths):
        post = post[:l, :]
        np.testing.assert_allclose(post.detach().numpy(), explicit_trellises_to_dense(gold), rtol=1e-6)


def test_posterior_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    cd, _ = make_constrained(i)

    p1 = cd.posterior(i1, l1)
    p2 = cd.posterior(i2, l2)

    batched = cd.posterior(i, l)

    one_x_one = torch.zeros((2, p1.shape[1], p1.shape[2]))
    one_x_one[0, :p1.shape[1]] = p1.squeeze(0)
    one_x_one[1, :p2.shape[1]] = p2.squeeze(0)

    np.testing.assert_allclose(batched.detach().numpy(), one_x_one.detach().numpy(), rtol=1e-6)


def test_posterior_decode_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    cd, _ = make_constrained(i)

    p1, s1 = cd.posterior_decode(i1, l1)
    p2, s2 = cd.posterior_decode(i2, l2)

    batched_p, batched_s = cd.posterior_decode(i, l)

    one_x_one_s = torch.cat([s1, s2], dim=0)

    one_x_one_p = torch.zeros((2, p1.shape[1]), dtype=p1.dtype)
    one_x_one_p[0, :p1.shape[1]] = p1.squeeze(0)
    one_x_one_p[1, :p2.shape[1]] = p2.squeeze(0)

    np.testing.assert_allclose(batched_s.detach().numpy(), one_x_one_s.detach().numpy(), rtol=1e-6)
    np.testing.assert_allclose(batched_p.detach().numpy(), one_x_one_p.detach().numpy(), rtol=1e-6)


def test_decode_batch_stable(generate_examples_and_batch):
    i1, _, l1, i2, _, l2, i, _, l = generate_examples_and_batch
    cd, _ = make_constrained(i)

    p1, s1 = cd.decode(i1, l1)
    p2, s2 = cd.decode(i2, l2)
    pad = torch.zeros((1, p1.size(1) - p2.size(1)), dtype=torch.long)
    one_x_one_p = torch.cat([p1, torch.cat([p2, pad], dim=1)], dim=0)
    one_x_one_s = torch.cat([s1, s2], dim=0)
    batched_p, batched_s =  cd.decode(i, l)
    np.testing.assert_allclose(one_x_one_s.detach().numpy(), batched_s.detach().numpy())
    for p1, p2 in zip(one_x_one_p, batched_p):
        np.testing.assert_allclose(p1.detach().numpy(), p2.detach().numpy())


def test_decode_shape(generate_batch):
    unary, _, lengths = generate_batch
    cd, _ = make_constrained(unary)

    paths, scores = cd.decode(unary, lengths)

    assert scores.shape == torch.Size([unary.size(0)])
    assert paths.shape == torch.Size([unary.size(0), unary.size(1)])

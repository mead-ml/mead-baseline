import os
import math
import json
import tempfile
from operator import itemgetter
import pytest
import numpy as np
from mock import patch, MagicMock
from eight_mile.utils import get_version, Offsets, to_numpy
tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.skipif(get_version(tf) < 2, reason="TF1.X")
from eight_mile.tf.layers import (
    CRF,
    transition_mask,
    StructuredTaggerDecoder,
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
    build_emission,
    build_trans as build_trans_,
    generate_batch as make_batch,
    generate_examples_and_batch as make_examples_and_batch,
)


def build_trans(t):
    return build_trans_(t.numpy().T)


@pytest.fixture
def generate_batch():
    """Generate a batch of data.

    Creates lengths such that at least one half of the batch is the maximum
    length.

    :returns: unary [B, T, H], tags [T, B], lengths [B]
    """
    scores, tags, lengths = map(tf.convert_to_tensor, make_batch())
    tags = tf.cast(tags, tf.int32)
    lengths = tf.cast(lengths, tf.int32)
    return scores, tags, lengths


@pytest.fixture
def generate_examples_and_batch():
    """A good test for these systems are do they produce the same results for a
    batch of data as when you feed the example in one by one.

    This function generates two single examples and then batches them together.
    """
    i1, t1, l1, i2, t2, l2, items, ts, lengths = map(tf.convert_to_tensor, make_examples_and_batch())
    t1 = tf.cast(t1, tf.int32)
    t2 = tf.cast(t2, tf.int32)
    ts = tf.cast(ts, tf.int32)
    l1 = tf.cast(l1, tf.int32)
    l2 = tf.cast(l2, tf.int32)
    lengths = tf.cast(lengths, tf.int32)
    return i1, t1, l1, i2, t2, l2, items, ts, lengths


def make_crf(unary):
    h = unary.shape[2]
    crf = CRF(h)
    crf.build([])
    trans = tf.random.uniform(shape=(h, h))
    crf.A = trans
    crf.fwd_layer.cell.set_weights([trans])
    crf.bwd_layer.cell.set_weights([trans])
    return crf, trans


def test_score_sentence(generate_batch):
    unary, tags, lengths = generate_batch
    tags = to_numpy(tags)
    crf, trans = make_crf(unary)

    sentence_score = crf.score_sentence(unary, tags, lengths)

    new_trans = build_trans(trans)
    scores = []
    for u, t, l in zip(unary, tags, lengths):
        emiss = build_emission(u[:l])
        golds = t[:l].tolist()
        scores.append(explicit_score_gold(emiss, new_trans, golds, Offsets.GO, Offsets.EOS))
    gold_scores = np.array(scores)
    np.testing.assert_allclose(sentence_score.numpy(), gold_scores, rtol=1e-6)


def test_score_sentence_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    score1 = crf.score_sentence(i1, t1, l1)
    score2 = crf.score_sentence(i2, t2, l2)
    one_x_one = tf.concat([score1, score2], axis=0)
    batched = crf.score_sentence(i, t, l)
    np.testing.assert_allclose(one_x_one.numpy(), batched.numpy())


def test_score_sentence_shape(generate_batch):
    unary, tags, lengths = generate_batch
    crf, _ = make_crf(unary)

    score = crf.score_sentence(unary, tags, lengths)
    assert score.shape == unary.shape[0]


def test_neg_log_loss(generate_batch):
   unary, tags, lengths = generate_batch
   crf, trans = make_crf(unary)

   nll = crf.neg_log_loss(unary, tags, lengths)

   new_trans = build_trans(trans)
   tags = to_numpy(tags)
   scores = []
   for u, t, l in zip(unary, tags, lengths):
       emiss = build_emission(u[:l])
       golds = t[:l].tolist()
       scores.append(explicit_nll(emiss, new_trans, golds, Offsets.GO, Offsets.EOS))
   gold_scores = np.mean(np.array(scores))
   np.testing.assert_allclose(nll.numpy(), gold_scores, rtol=1e-6)


def test_neg_log_loss_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    nll1 = crf.neg_log_loss(i1, t1, l1)
    nll2 = crf.neg_log_loss(i2, t2, l2)
    one_x_one = (nll1 + nll2) / 2
    batched = crf.neg_log_loss(i, t, l)
    np.testing.assert_allclose(one_x_one.numpy(), batched.numpy())


def test_partition(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    partition = crf.partition(unary, lengths)

    new_trans = build_trans(trans)
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        scores.append(explicit_forward(emiss, new_trans, Offsets.GO, Offsets.EOS)[0])
    gold_scores = np.array(scores)
    np.testing.assert_allclose(partition.numpy(), gold_scores, rtol=1e-6)


def test_partition_batch_stable(generate_examples_and_batch):
    i1, _, l1, i2, _, l2, i, _, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    part1 = crf.partition(i1, l1)
    part2 = crf.partition(i2, l2)
    one_x_one = tf.concat([part1, part2], axis=0)
    batched = crf.partition(i, l)
    np.testing.assert_allclose(one_x_one.numpy(), batched.numpy())


def test_partition_shape(generate_batch):
    unary, _, lengths = generate_batch
    crf, _ = make_crf(unary)
    fwd = crf.partition(unary, lengths)
    assert fwd.shape == unary.shape[0]


def test_partition_over_time(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    forward = crf.partition_over_time(unary, lengths)

    new_trans = build_trans(trans)
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        scores.append(explicit_forward(emiss, new_trans, Offsets.GO, Offsets.EOS)[1])
    for fwd, gold, l in zip(forward, scores, lengths):
        fwd = fwd[:l, :]
        np.testing.assert_allclose(fwd.numpy(), explicit_trellises_to_dense(gold), rtol=1e-6)


def test_forward_over_time_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    f1 = crf.partition_over_time(i1, l1)
    f2 = crf.partition_over_time(i2, l2)

    batched = crf.partition_over_time(i, l)

    one_x_one = np.zeros((2, f1.shape[1], f1.shape[2]))
    one_x_one[0, :f1.shape[1]] = f1.numpy().squeeze(0)
    one_x_one[1, :f2.shape[1]] = f2.numpy().squeeze(0)
    one_x_one = one_x_one

    np.testing.assert_allclose(one_x_one, batched.numpy(), rtol=1e-6)


def test_backward_over_time(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    backward = crf.partition_backward_over_time(unary, lengths)

    new_trans = build_trans(trans)
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        scores.append(explicit_backward(emiss, new_trans, Offsets.GO, Offsets.EOS)[1])
    for bwd, gold, l in zip(backward, scores, lengths):
        bwd = bwd[:l, :]
        np.testing.assert_allclose(bwd.numpy(), explicit_trellises_to_dense(gold), rtol=1e-6)


def test_backward_over_time_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    part1 = crf.partition_backward_over_time(i1, l1)
    part2 = crf.partition_backward_over_time(i2, l2)

    batched = crf.partition_backward_over_time(i, l)

    one_x_one = np.zeros((2, part1.shape[1], part1.shape[2]))
    one_x_one[0, :part1.shape[1]] = part1.numpy().squeeze(0)
    one_x_one[1, :part2.shape[1]] = part2.numpy().squeeze(0)

    np.testing.assert_allclose(one_x_one, batched.numpy(), rtol=1e-6, atol=1e-6)


def test_posterior(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    posterior = crf.posterior(unary, lengths)

    new_trans = build_trans(trans)
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        scores.append(explicit_posterior(emiss, new_trans, Offsets.GO, Offsets.EOS))
    for post, gold, l in zip(posterior, scores, lengths):
        post = post[:l, :]
        np.testing.assert_allclose(post.numpy(), explicit_trellises_to_dense(gold), rtol=1e-6)


def test_posterior_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    p1 = crf.posterior(i1, l1)
    p2 = crf.posterior(i2, l2)

    batched = crf.posterior(i, l)

    one_x_one = np.zeros((2, p1.shape[1], p1.shape[2]))
    one_x_one[0, :p1.shape[1]] = p1.numpy().squeeze(0)
    one_x_one[1, :p2.shape[1]] = p2.numpy().squeeze(0)

    np.testing.assert_allclose(batched.numpy(), one_x_one, rtol=1e-6)


def test_posterior_decode(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    paths, scores = crf.posterior_decode(unary, lengths)

    new_trans = build_trans(trans)
    gold_paths = []
    gold_scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        p, s = explicit_posterior_decode(emiss, new_trans, Offsets.GO, Offsets.EOS)
        gold_paths.append(p)
        gold_scores.append(s)
    for p, g, l in zip(paths, gold_paths, lengths):
        p = p[:l]
        np.testing.assert_allclose(p.numpy(), np.array(g), rtol=1e-6)
    np.testing.assert_allclose(scores.numpy(), np.array(gold_scores), rtol=1e-6)


def test_posterior_decode_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    p1, s1 = crf.posterior_decode(i1, l1)
    p2, s2 = crf.posterior_decode(i2, l2)

    batched_p, batched_s = crf.posterior_decode(i, l)

    one_x_one_s = tf.concat([s1, s2], axis=0)

    one_x_one_p = np.zeros((2, p1.shape[1]), np.int)
    one_x_one_p[0, :p1.shape[1]] = p1.numpy().squeeze(0)
    one_x_one_p[1, :p2.shape[1]] = p2.numpy().squeeze(0)

    np.testing.assert_allclose(batched_s.numpy(), one_x_one_s.numpy(), rtol=1e-6)
    np.testing.assert_allclose(batched_p.numpy(), one_x_one_p, rtol=1e-6)

def test_viterbi(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    tf_path, tf_scores = crf.decode(unary, lengths)

    new_trans = build_trans(trans)
    paths = []
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        p, s = explicit_viterbi(emiss, new_trans, Offsets.GO, Offsets.EOS)
        scores.append(s)
        paths.append(p)
    gold_scores = np.array(scores)
    np.testing.assert_allclose(tf_scores.numpy(), gold_scores, rtol=1e-6)
    for pp, l, p in zip(tf_path, lengths, paths):
        assert pp[:l].numpy().tolist() == p

def test_decode_batch_stable(generate_examples_and_batch):
    i1, _, l1, i2, _, l2, i, _, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    p1, s1 = crf.decode(i1, l1)
    p2, s2 = crf.decode(i2, l2)
    pad = tf.zeros((1, p1.shape[1] - p2.shape[1]), dtype=tf.int32)
    one_x_one_p = tf.concat([p1, tf.concat([p2, pad], axis=1)], axis=0)
    one_x_one_s = tf.concat([s1, s2], axis=0)
    batched_p, batched_s =  crf.decode(i, l)
    np.testing.assert_allclose(one_x_one_s.numpy(), batched_s.numpy())
    for p1, p2 in zip(one_x_one_p, batched_p):
        np.testing.assert_allclose(p1.numpy(), p2.numpy())


def test_decode_shape(generate_batch):
    unary, _, lengths = generate_batch
    crf, _ = make_crf(unary)

    paths, scores = crf.decode(unary, lengths)

    assert scores.shape == unary.shape[0]
    assert paths.shape == unary.shape[:2]


def test_add_states_to_tags():
    B = np.random.randint(5, 11)
    T = np.random.randint(12, 22)
    H = np.random.randint(23, 45)
    lengths = np.random.randint(1, T, size=(B,)).astype(np.int32)
    lengths[np.random.randint(0, B, size=(B // 2,))] = T

    tags = np.random.randint(1, H, size=(B, T))

    for t, l in zip(tags, lengths):
        t[l:] = 0

    g_tags = []
    for t, l in zip(tags, lengths):
        g_tags.append(np.array([Offsets.GO] + t[:l].tolist() + [Offsets.EOS]))

    gold_tags = np.zeros((B, T + 2), dtype=np.int32)

    for i, g in enumerate(g_tags):
        gold_tags[i, :len(g)] = g

    tags = StructuredTaggerDecoder._add_states_to_tags(tags, lengths)

    np.testing.assert_allclose(tags.numpy(), gold_tags)


def test_add_states_to_unary():
    B = np.random.randint(5, 11)
    T = np.random.randint(12, 22)
    H = np.random.randint(23, 45)
    lengths = np.random.randint(1, T, size=(B,)).astype(np.int32)
    lengths[np.random.randint(0, B, size=(B // 2,))] = T

    unary = np.random.rand(B, T, H).astype(np.float32)

    for u, l in zip(unary, lengths):
        u[l:, :] = 0

    us = []
    start = np.full((1, H), -1e4)
    start[:, Offsets.GO] = 0
    end = np.full((1, H), -1e4)
    end[:, Offsets.EOS] = 0
    for u, l in zip(unary, lengths):
        us.append(np.concatenate([start, u[:l], end], axis=0))

    gold_unary = np.zeros((B, T + 2, H))
    for i, g in enumerate(us):
        gold_unary[i, :len(g), :] = g

    unary = StructuredTaggerDecoder._add_states_to_unary(unary, lengths)

    np.testing.assert_allclose(unary.numpy(), gold_unary)

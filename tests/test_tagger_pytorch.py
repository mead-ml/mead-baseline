#!/usr/bin/env python3


import math
import pytest
import numpy as np
torch = pytest.importorskip("torch")
from eight_mile.pytorch.layers import (
    GreedyTaggerDecoder,
)
from tagger_decode_utils import (
    build_emission,
    explicit_sparse_cross_entropy,
    explicit_softmax,
    generate_batch as make_batch,
    generate_examples_and_batch as make_examples_and_batch,
)


@pytest.fixture
def generate_batch():
    return [x for x in map(torch.from_numpy, make_batch())]


@pytest.fixture
def generate_examples_and_batch():
    i1, t1, l1, i2, t2, l2, items, tags, lengths = map(torch.from_numpy, make_examples_and_batch())
    return i1, t1, l1, i2, t2, l2, items, tags, lengths


def test_neg_log_loss(generate_batch):
    unary, tags, lengths = generate_batch
    tagger = GreedyTaggerDecoder(unary.size(-1))

    nll = tagger.neg_log_loss(unary, tags, lengths)

    scores = []
    for u, t, l in zip(unary, tags, lengths):
        emiss = build_emission(u[:l])
        golds = t[:l].tolist()
        scores.append(explicit_sparse_cross_entropy(emiss, golds))
    gold_scores = np.mean(np.array(scores))
    np.testing.assert_allclose(nll.detach().numpy(), gold_scores, rtol=1e-6)


def test_neg_log_loss_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, items, tags, lengths = generate_examples_and_batch
    tagger = GreedyTaggerDecoder(items.size(-1))

    nll1 = tagger.neg_log_loss(i1, t1, l1)
    nll2 = tagger.neg_log_loss(i2, t2, l2)
    one_x_one = (nll1 + nll2) / 2
    batched = tagger.neg_log_loss(items, tags, lengths)

    np.testing.assert_allclose(one_x_one.detach().numpy(), batched.detach().numpy(), rtol=1e-6)


def test_score_sentence(generate_batch):
    unary, tags, lengths = generate_batch
    tagger = GreedyTaggerDecoder(unary.size(-1))

    scored = tagger.score_sentence(unary, tags, lengths)

    scores = []
    for u, t, l in zip(unary, tags, lengths):
        emiss = build_emission(u[:l])
        golds = t[:l].tolist()
        scores.append(math.fsum(explicit_softmax(e)[t] for e, t in zip(emiss, golds)))
    np.testing.assert_allclose(scored.detach().numpy(), np.array(scores), rtol=1e-6)


def test_score_sentence_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, items, tags, lengths = generate_examples_and_batch
    tagger = GreedyTaggerDecoder(items.size(-1))

    score1 = tagger.score_sentence(i1, t1, l1)
    score2 = tagger.score_sentence(i2, t2, l2)
    one_x_one = torch.cat([score1, score2])
    batched = tagger.score_sentence(items, tags, lengths)

    np.testing.assert_allclose(one_x_one.detach().numpy(), batched.detach().numpy(), rtol=1e-6)


def test_posterior(generate_batch):
    unary, tags, lengths = generate_batch
    tagger = GreedyTaggerDecoder(unary.size(-1))
    posterior = tagger.posterior(unary, lengths)

    post = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        post.append([[explicit_softmax(e)[i] for i in range(len(e))] for e in emiss])
    mx = max(len(p) for p in post)
    gold_post = np.zeros((len(post), mx, unary.size(-1)), dtype=np.float32)
    print(gold_post.shape)
    for i, p in enumerate(post):
        gold_post[i, :len(p), :] = np.array(p)
    np.testing.assert_allclose(posterior.detach().numpy(), gold_post, rtol=1e-6)


def test_posterior_batch_stable(generate_examples_and_batch):
    i1, _, l1, i2, _, l2, items, tags, lengths = generate_examples_and_batch
    tagger = GreedyTaggerDecoder(items.size(-1))

    post1 = tagger.posterior(i1, l1)
    post2 = tagger.posterior(i2, l2)
    one_x_one = torch.zeros((2, max(post1.size(1), post2.size(1)), items.size(-1)))
    one_x_one[0, :post1.size(1), :] = post1
    one_x_one[1, :post2.size(1), :] = post2
    batched = tagger.posterior(items, lengths)

    np.testing.assert_allclose(one_x_one.detach().numpy(), batched.detach().numpy(), rtol=1e-6)


def test_decode(generate_batch):
    unary, _, lengths = generate_batch
    tagger = GreedyTaggerDecoder(unary.size(-1))

    paths, scores = tagger.decode(unary, lengths)

    gold_paths = []
    gold_scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        path = []
        score = 0
        for e in emiss:
            path.append(max(e, key=lambda x: e[x]))
            score += explicit_softmax(e)[path[-1]]
        gold_paths.append(path)
        gold_scores.append(score)

    for p, gp, l in zip(paths, gold_paths, lengths):
        np.testing.assert_equal(p[:l].detach().numpy(), np.array(gp))
    for s, gs in zip(scores, gold_scores):
        np.testing.assert_allclose(s.detach().numpy(), gs, rtol=1e-6)


def test_decode_batch_stable(generate_examples_and_batch):
    i1, _, l1, i2, _, l2, items, tags, lengths = generate_examples_and_batch
    tagger = GreedyTaggerDecoder(items.size(-1))

    path1, score1 = tagger.decode(i1, l1)
    path2, score2 = tagger.decode(i2, l2)
    path_one_x_one = torch.zeros((2, max(path1.size(1), path2.size(1))))
    path_one_x_one[0, :path1.size(1)] = path1
    path_one_x_one[1, :path2.size(1)] = path2
    score_one_x_one = torch.cat([score1, score2])
    batched_path, batched_score = tagger.decode(items, lengths)

    np.testing.assert_allclose(path_one_x_one.detach().numpy(), batched_path.detach().numpy(), rtol=1e-6)
    np.testing.assert_allclose(score_one_x_one.detach().numpy(), batched_score.detach().numpy(), rtol=1e-6)

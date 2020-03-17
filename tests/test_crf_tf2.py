import pytest
import numpy as np
from eight_mile.utils import get_version, Offsets

tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.skipif(get_version(tf) < 2, reason="TF1.X")
from eight_mile.tf.layers import (
    CRF,
    StructuredTaggerDecoder,
    get_shape_as_list,
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
    build_trans as make_trans,
    build_emission as make_emission,
    generate_batch as make_batch,
    generate_examples_and_batch as make_examples_and_batch,
)


@pytest.fixture
def generate_batch():
    scores, tags, lengths = make_batch()
    scores = tf.convert_to_tensor(scores)
    tags = tf.convert_to_tensor(tags)
    lengths = tf.convert_to_tensor(lengths, dtype=tf.int32)
    return scores, tags, lengths


@pytest.fixture
def generate_examples_and_batch():
    i1, t1, l1, i2, t2, l2, items, ts, lengths = map(tf.convert_to_tensor, make_examples_and_batch())
    t1 = tf.cast(t1, tf.int32)
    t2 = tf.cast(t2, tf.int32)
    ts = tf.cast(ts, tf.int32)
    l1 = tf.cast(l1, tf.int32)
    l2 = tf.cast(l2, tf.int32)
    lengths = tf.cast(lengths, tf.int32)
    return i1, t1, l1, i2, t2, l2, items, ts, lengths


def make_crf(unary, zero_trans = False):
    h = get_shape_as_list(unary)[-1]
    crf = CRF(h)
    crf.build(unary.shape)
    if not zero_trans:
        trans = tf.random.uniform((h, h))
        # Assign to the variable so the updated value get used by the fwd/bwd layers
        crf.A.assign(trans)
    return crf, crf.A


def build_trans(trans):
    # The code that consumes this transition dict use (to, from) like pyt while the tf uses (from, to) so transpose so the numbers work
    return make_trans(np.transpose(trans.numpy()))

def build_emission(emission):
    return make_emission(emission.numpy())


def test_score_sentence(generate_batch):
    unary, tags, lengths = generate_batch
    crf, trans = make_crf(unary)

    score = crf.score_sentence(unary, tags, lengths)

    new_trans = build_trans(trans)
    tags = tags.numpy()
    scores = []
    for u, t, l in zip(unary, tags, lengths):
        emiss = build_emission(u[:l])
        golds = t[:l].tolist()
        scores.append(explicit_score_gold(emiss, new_trans, golds, Offsets.GO, Offsets.EOS))
    gold_scores = np.array(scores)
    np.testing.assert_allclose(score.numpy(), gold_scores, rtol=1e-6)


def test_score_sentence_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    score1 = crf.score_sentence(i1, t1, l1)
    score2 = crf.score_sentence(i2, t2, l2)
    one_x_one = tf.concat([score1, score2], axis=0)

    batched = crf.score_sentence(i, t, l)

    np.testing.assert_allclose(one_x_one.numpy(), batched.numpy(), rtol=1e-6)


def test_score_sentence_shape(generate_batch):
    unary, tags, lengths = generate_batch
    b, *_, h = get_shape_as_list(unary)
    crf = CRF(h)
    crf.build(unary.shape)
    score = crf.score_sentence(unary, tags, lengths)
    assert score.shape == b


def test_neg_log_loss(generate_batch):
    unary, tags, lengths = generate_batch
    crf, trans = make_crf(unary)

    nll = crf.neg_log_loss(unary, tags, lengths)

    new_trans = build_trans(trans)
    tags = tags.numpy()
    scores = []
    for u, t, l in zip(unary, tags, lengths):
        golds = t[:l].tolist()
        emiss = build_emission(u[:l])
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


def test_forward(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    fwd = crf.partition(unary, lengths)

    new_trans = build_trans(trans)
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        scores.append(explicit_forward(emiss, new_trans, Offsets.GO, Offsets.EOS)[0])
    gold_scores = np.array(scores)
    np.testing.assert_allclose(fwd.numpy(), gold_scores, rtol=1e-6)


def test_forward_batch_stable(generate_examples_and_batch):
    i1, _, l1, i2, _, l2, i, _, l = generate_examples_and_batch
    crf, trans = make_crf(i)

    fwd1 = crf.partition(i1, l1)
    fwd2 = crf.partition(i2, l2)
    one_x_one = tf.concat([fwd1, fwd2], axis=0)

    batched = crf.partition(i, l)

    np.testing.assert_allclose(one_x_one.numpy(), batched.numpy(), rtol=1e-6)


def test_forward_batch_stable(generate_examples_and_batch):
    i1, _, l1, i2, _, l2, i, _, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    fw1 = crf.partition(i1, l1)
    fw2 = crf.partition(i2, l2)
    one_x_one = tf.concat([fw1, fw2], axis=0)
    batched = crf.partition(i, l)
    np.testing.assert_allclose(one_x_one.numpy(), batched.numpy())


def test_forward_shape(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)
    fwd = crf.partition(unary, lengths)
    assert fwd.shape == get_shape_as_list(unary)[0]


def test_forward_over_time(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    fwd = crf.partition_over_time(unary, lengths)

    new_trans = build_trans(trans)
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        scores.append(explicit_forward(emiss, new_trans, Offsets.GO, Offsets.EOS)[1])
    for f, g, l in zip(fwd, scores, lengths):
        f = f[:l, :]
        np.testing.assert_allclose(f.numpy(), explicit_trellises_to_dense(g), rtol=1e-6)


def test_forward_over_time_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    f1 = crf.partition_over_time(i1, l1)
    f2 = crf.partition_over_time(i2, l2)

    batched = crf.partition_over_time(i, l)

    _, t, h = get_shape_as_list(f1)
    one_x_one = np.zeros((2, t, h))
    one_x_one[0, :t] = tf.squeeze(f1, 0).numpy()
    one_x_one[1, :get_shape_as_list(f2)[1]] = tf.squeeze(f2, 0).numpy()

    np.testing.assert_allclose(one_x_one, batched.numpy(), rtol=1e-6)


def test_backward_over_time(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    bwd = crf.partition_backward_over_time(unary, lengths)

    new_trans = build_trans(trans)
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        scores.append(explicit_backward(emiss, new_trans, Offsets.GO, Offsets.EOS)[1])
    for b, g, l in zip(bwd, scores, lengths):
        b = b[:l, :]
        np.testing.assert_allclose(b.numpy(), explicit_trellises_to_dense(g), rtol=1e-6)


def test_backward_over_time_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    f1 = crf.partition_backward_over_time(i1, l1)
    f2 = crf.partition_backward_over_time(i2, l2)

    batched = crf.partition_backward_over_time(i, l)

    _, t, h = get_shape_as_list(f1)
    one_x_one = np.zeros((2, t, h))
    one_x_one[0, :t] = tf.squeeze(f1, 0).numpy()
    one_x_one[1, :get_shape_as_list(f2)[1]] = tf.squeeze(f2, 0).numpy()

    np.testing.assert_allclose(one_x_one, batched.numpy(), rtol=1e-6)


def test_posterior(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    posterior = crf.posterior(unary, lengths)

    new_trans = build_trans(trans)
    scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        scores.append(explicit_posterior(emiss, new_trans, Offsets.GO, Offsets.EOS))
    for p, g, l in zip(posterior, scores, lengths):
        p = p[:l, :]
        np.testing.assert_allclose(p.numpy(), explicit_trellises_to_dense(g), rtol=1e-6)


def test_posterior_batch_stable(generate_examples_and_batch):
    i1, t1, l1, i2, t2, l2, i, t, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    p1 = crf.posterior(i1, l1)
    p2 = crf.posterior(i2, l2)

    batched = crf.posterior(i, l)

    _, t, h = get_shape_as_list(p1)
    t2 = get_shape_as_list(p2)[1]
    one_x_one = np.zeros((2, t, h))
    one_x_one[0, :t] = tf.squeeze(p1, 0).numpy()
    one_x_one[1, :t2] = tf.squeeze(p2, 0).numpy()

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

    _, t = get_shape_as_list(p1)
    t2 = get_shape_as_list(p2)[1]
    one_x_one_p = np.zeros((2, t), dtype=np.int32)
    one_x_one_p[0, :t] = tf.squeeze(p1, 0).numpy()
    one_x_one_p[1, :t2] = tf.squeeze(p2, 0).numpy()

    np.testing.assert_allclose(batched_s.numpy(), one_x_one_s.numpy(), rtol=1e-6)
    np.testing.assert_allclose(batched_p.numpy(), one_x_one_p, rtol=1e-6)


def test_decode(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    path, scores = crf.decode(unary, lengths)

    new_trans = build_trans(trans)
    gold_paths = []
    gold_scores = []
    for u, l in zip(unary, lengths):
        emiss = build_emission(u[:l])
        p, s = explicit_viterbi(emiss, new_trans, Offsets.GO, Offsets.EOS)
        gold_scores.append(s)
        gold_paths.append(p)
    gold_scores = np.array(scores)
    np.testing.assert_allclose(scores.numpy(), gold_scores, rtol=1e-6)
    for p, g, l in zip(path, gold_paths, lengths):
        assert p[:l].numpy().tolist() == g


def test_decode_batch_stable(generate_examples_and_batch):
    i1, _, l1, i2, _, l2, i, _, l = generate_examples_and_batch
    crf, _ = make_crf(i)

    p1, s1 = crf.decode(i1, l1)
    p2, s2 = crf.decode(i2, l2)

    batched_p, batched_s = crf.decode(i, l)

    one_x_one_s = tf.concat([s1, s2], axis=0)

    _, t = get_shape_as_list(p1)
    t2 = get_shape_as_list(p2)[1]
    one_x_one_p = np.zeros((2, t), dtype=np.int32)
    one_x_one_p[0, :t] = tf.squeeze(p1, 0).numpy()
    one_x_one_p[1, :t2] = tf.squeeze(p2, 0).numpy()

    np.testing.assert_allclose(one_x_one_p, batched_p.numpy(), rtol=1e-6)
    np.testing.assert_allclose(one_x_one_s.numpy(), batched_s.numpy(), rtol=1e-6)


def test_decode_shape(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    paths, scores = crf.decode(unary, lengths)

    b, t, *_ = get_shape_as_list(unary)
    assert scores.shape == (b,)
    assert paths.shape == (b, t)


def test_viterbi_scores_equal_score_sentence(generate_batch):
    unary, _, lengths = generate_batch
    crf, trans = make_crf(unary)

    p, viterbi_scores = crf.decode(unary, lengths)
    gold_scores = crf.score_sentence(unary, p, lengths)

    np.testing.assert_allclose(viterbi_scores.numpy(), gold_scores.numpy(), rtol=1e-6)


def test_viterbi_degrates_to_argmax(generate_batch):
    unary, _, lengths = generate_batch

    crf, trans = make_crf(unary, zero_trans=True)

    path, score = crf.decode(unary, lengths)

    gold_path = tf.argmax(unary, axis=-1).numpy()
    gold_score = tf.reduce_max(unary, axis=-1).numpy()

    for i, l in enumerate(lengths):
        gold_path[i, l:] = 0
        gold_score[i, l:] = 0

    gold_score = np.sum(gold_score, 1)

    np.testing.assert_allclose(path.numpy(), gold_path)
    np.testing.assert_allclose(score.numpy(), gold_score, rtol=1e-6)


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

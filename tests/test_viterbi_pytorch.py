#!/usr/bin/env python3


import tempfile
import pytest
import numpy as np
torch = pytest.importorskip("torch")
from eight_mile.utils import Offsets, get_version
from eight_mile.pytorch.layers import (
    Viterbi,
    script_viterbi,
    ViterbiBatchSize1,
    CRF,
    ConstrainedGreedyTaggerDecoder,
)
from tagger_decode_utils import (
    build_trans,
    build_emission,
    explicit_viterbi,
    generate_batch as make_batch,
    generate_examples_and_batch as make_examples_and_batch,
)

TRIALS = 100

@pytest.fixture
def generate_batch():
    scores, tags, lengths = map(torch.from_numpy, make_batch())
    return scores.transpose(0, 1), tags.transpose(0, 1), lengths


@pytest.fixture
def generate_examples_and_batch():
    i1, t1, l1, i2, t2, l2, items, tags, lengths = map(torch.from_numpy, make_examples_and_batch())
    i1, t1, i2, t2, items, tags = map(lambda x: x.transpose(0, 1), [i1, t1, i2, t2, items, tags])
    return i1, t1, l1, i2, t2, l2, items, tags, lengths


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


def test_viterbi_shape(generate_batch):
    unary, _, lengths = generate_batch
    h = unary.size(2)
    trans = torch.rand(1, h, h)
    viterbi = Viterbi(Offsets.GO, Offsets.EOS)
    paths, scores = viterbi(unary, trans, lengths)
    assert scores.shape == torch.Size([unary.size(1)])
    assert paths.shape == torch.Size([unary.size(0), unary.size(1)])


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


@pytest.mark.skipif(get_version(torch) <= 1.4, reason="Old ONNX")
def test_ONNX_export():
    ort = pytest.importorskip("onnxruntime")

    v = ViterbiBatchSize1(Offsets.GO, Offsets.EOS)

    B = 1
    T = np.random.randint(10, 100)
    H = np.random.randint(24, 76)

    unary = torch.rand(T, B, H)
    trans = torch.rand(1, H, H)
    length = torch.randint(1, T, size=(B,))

    p1, s1 = v(unary, trans, length)

    with tempfile.NamedTemporaryFile() as f:
        torch.onnx.export(
            v,
            (unary, trans, length),
            verbose=True,
            dynamic_axes={"path": {0: "sequence"}, "unary": {0: "sequence"}},
            f=f.name,
            input_names=["unary", "trans", "length"],
            output_names=["path", "score"],
            opset_version=12,
            example_outputs=[p1, s1],
        )
        o = ort.InferenceSession(f.name)

        for _ in range(TRIALS):
            T = np.random.randint(10, 100)
            unary = np.random.rand(T, B, H).astype(np.float32)
            trans = np.random.rand(1, H, H).astype(np.float32)
            length = np.random.randint(1, T, size=(B,)).astype(np.int64)

            p1, s1 = v(*tuple(map(torch.from_numpy, (unary, trans, length))))
            p1 = p1.numpy()
            s1 = s1.numpy()

            inputs = {"unary": unary, "trans": trans, "length": length}
            p2, s2 = o.run(None, {k: v for k, v in inputs.items() if k in set(i.name for i in o.get_inputs())})

            np.testing.assert_allclose(p1, p2, atol=1e-6)
            np.testing.assert_allclose(s1, s2, atol=1e-6)


def test_viterbi_score_equals_sentence_score_crf(generate_batch):
    """Test that the scores from viterbi decoding are the same scores that you get when looking up those returned paths."""
    unary, _, lengths = generate_batch
    h = unary.size(2)
    crf = CRF(h, batch_first=False)
    trans = torch.rand(h, h)
    crf.transitions_p.data = trans.unsqueeze(0)

    p, viterbi_scores = Viterbi(Offsets.GO, Offsets.EOS)(unary, crf.transitions, lengths)
    gold_scores = crf.score_sentence(unary, p, lengths)
    np.testing.assert_allclose(viterbi_scores.detach().numpy(), gold_scores.detach().numpy(), rtol=1e-6)


def test_viterbi_score_equals_sentence_score_cd(generate_batch):
    """Test that the scores from viterbi decoding are the same scores that you get when looking up those returned paths."""
    unary, _, lengths = generate_batch
    h = unary.size(2)
    constraint_mask = torch.randint(0, 1, (1, h, h)).to(torch.bool)
    cd = ConstrainedGreedyTaggerDecoder(h, constraint_mask, batch_first=False)

    p, viterbi_scores = Viterbi(Offsets.GO, Offsets.EOS)(unary, cd.transitions, lengths)
    gold_scores = cd.score_sentence(unary, p, lengths)
    np.testing.assert_allclose(viterbi_scores.detach().numpy(), gold_scores.detach().numpy(), rtol=1e-6)

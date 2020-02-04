import os
import math
import random
import string
from functools import partial
import pytest
import numpy as np
from eight_mile.embeddings import *
from eight_mile.utils import Offsets

loc = os.path.dirname(os.path.realpath(__file__))
GLOVE_FILE = os.path.join(loc, "test_data", "glove_test.txt")
W2V_FILE = os.path.join(loc, "test_data", "w2v_test.bin")


def random_model():
    """Randomly pick a model for testing."""
    files = [GLOVE_FILE, W2V_FILE]
    f = random.choice(files)
    return partial(PretrainedEmbeddingsModel, f)


def test_glove_vsz():
    # Demo data = 10
    gold_vsz = 10 + len(Offsets.VALUES)
    wv = PretrainedEmbeddingsModel(GLOVE_FILE, keep_unused=True)
    assert wv.get_vsz() == gold_vsz


def test_glove_dsz():
    gold_dsz = 50
    wv = PretrainedEmbeddingsModel(GLOVE_FILE, keep_unused=True)
    assert wv.get_dsz() == gold_dsz


def test_w2v_vsz():
    gold_vsz = 10 + len(Offsets.VALUES)
    wv = PretrainedEmbeddingsModel(W2V_FILE, keep_unused=True)
    assert wv.get_vsz() == gold_vsz


def test_w2v_dsz():
    gold_dsz = 300
    wv = PretrainedEmbeddingsModel(W2V_FILE, keep_unused=True)
    assert wv.get_dsz() == gold_dsz


# def test_rand_vsz():
#     ref_wv = random_model()(keep_unused=True)
#     vocab = ref_wv.vocab
#     gold_dsz = ref_wv.get_dsz()
#     wv = RandomInitVecModel(gold_dsz, vocab)
#     assert wv.get_vsz() == len(vocab)


def test_rand_dsz():
    ref_wv = random_model()(keep_unused=True)
    vocab = ref_wv.vocab
    gold_dsz = ref_wv.get_dsz()
    wv = RandomInitVecModel(gold_dsz, vocab)
    assert wv.get_dsz() == gold_dsz


def test_random_vector_range():
    gold_weight = 0.4
    wv = RandomInitVecModel(300, {k: 1 for k in list(string.ascii_letters)}, unif_weight=gold_weight)
    assert np.min(wv.weights) >= -gold_weight
    assert np.max(wv.weights) <= gold_weight


def test_round_trip():
    gold_weight = 0.4
    input_model = RandomInitVecModel(300, {k: 1 for k in list(string.ascii_letters)}, unif_weight=gold_weight)
    write_word2vec_file("test.bin", input_model.vocab, input_model.weights)
    output_model = PretrainedEmbeddingsModel("test.bin", keep_unused=True)
    # This is a bit weird...
    assert output_model.vsz == input_model.vsz
    for word in input_model.vocab:
        if word not in Offsets.VALUES:
            assert np.allclose(input_model[word], output_model[word])


# def test_valid_lookup():
#     wv = random_model()(keep_unused=True)
#     key = '<PAD>'
#     while key == '<PAD>':
#         key = random.choice(list(wv.vocab.keys()))
#     res = wv.lookup(key)
#     assert res is not None
#     assert res.shape == (wv.get_dsz(),)
#     with pytest.raises(AssertionError):
#         np.testing.assert_allclose(res, np.zeros((wv.get_dsz(),)))


def test_nullv_lookup():
    wv = random_model()(keep_unused=True)
    key = "zzzzzzzzzzz"
    assert key not in wv.vocab
    res = wv.lookup(key, nullifabsent=False)
    np.testing.assert_allclose(res, np.zeros((wv.get_dsz())))


def test_none_lookup():
    wv = random_model()(keep_unused=True)
    key = "zzzzzzzzzzz"
    assert key not in wv.vocab
    res = wv.lookup(key)
    assert res is None


def test_mmap_glove():
    wv_file = PretrainedEmbeddingsModel(GLOVE_FILE, keep_unused=True)
    wv_mmap = PretrainedEmbeddingsModel(GLOVE_FILE, keep_unused=True, use_mmap=True)
    np.testing.assert_allclose(wv_file.weights, wv_mmap.weights)


def test_mmap_w2v():
    wv_file = PretrainedEmbeddingsModel(W2V_FILE, keep_unused=True)
    wv_mmap = PretrainedEmbeddingsModel(W2V_FILE, keep_unused=True, use_mmap=True)
    np.testing.assert_allclose(wv_file.weights, wv_mmap.weights)


def test_normalize_e2e():
    wv = random_model()(normalize=True, keep_unused=True)
    norms = np.sqrt(np.sum(np.square(wv.weights), 1))
    for norm in norms:
        assert norm == 0 or np.allclose(norm, 1, rtol=1e-4)


def test_normalize():
    wv = random_model()(keep_unused=True)
    normed = norm_weights(wv.weights)
    gold_norms = np.zeros_like(wv.weights)
    for i in range(len(gold_norms)):
        norm = np.sqrt(np.sum(np.square(wv.weights[i])))
        gold_norms[i] = wv.weights[i] if norm == 0.0 else wv.weights[i] / norm
    np.testing.assert_allclose(normed, gold_norms)


# def test_vocab_truncation():
#     model = random_model()
#     wv = model(keep_unused=True)
#     gold = wv.vocab
#     keys = list(gold.keys())
#     removed = '<PAD>'
#     while removed == '<PAD>':
#         removed = random.choice(keys)
#     gold.pop(removed)
#     wv = model(known_vocab=gold)
#     assert set(gold.keys()) == set(wv.vocab.keys())
#     assert removed not in wv.vocab


def test_vocab_not_truncated():
    model = random_model()
    wv = model(keep_unused=True)
    gold = wv.vocab
    keys = list(gold.keys())
    removed = "<PAD>"
    while removed == "<PAD>":
        removed = random.choice(keys)
    gold.pop(removed)
    assert set(keys) != set(gold)
    wv = model(known_vocab=gold, keep_unused=True)
    assert set(keys) == set(wv.vocab.keys())
    assert removed in wv.vocab


def test_extra_vocab():
    model = random_model()
    wv = model(keep_unused=True)
    vocab = wv.vocab
    extra_keys = ["AAAAAAAAAAAA", "ZZZZZZZZZZ"]
    for key in extra_keys:
        vocab[key] = 12
    wv = model(known_vocab=vocab)
    for key in extra_keys:
        assert key in wv.vocab
    for key in extra_keys:
        np.testing.assert_allclose(wv.lookup(key), np.zeros((wv.get_dsz(),)))


def test_extra_vocab_weights():
    """Test that words not in the pretrained vocab are initialized correctly."""
    weight = 0.5
    model = random_model()
    wv = model(keep_unused=True, unif_weight=weight)
    vocab = wv.vocab
    extra_keys = ["AAAAAAAAAAAA", "ZZZZZZZZZZ"]
    for key in extra_keys:
        vocab[key] = 12
    wv = model(known_vocab=vocab)
    for key in extra_keys:
        vec = wv.lookup(key)
        assert np.min(vec) >= -weight
        assert np.max(vec) <= weight


def test_rand_no_counts():
    """Test that unattested word are not removed without counts."""
    vocab = {"A": 0, "B": 0, "C": 13, "D": 55}
    wv = RandomInitVecModel(random.randint(1, 301), vocab, counts=False)
    assert wv.get_vsz() == len(vocab)


def test_rand_counts():
    """Test that unattested word are removed with counts."""
    vocab = {"A": 0, "B": 0, "C": 13, "D": 55}
    wv = RandomInitVecModel(12, vocab, counts=True)
    gold = {"C", "D"}
    for g in gold:
        assert g in wv.vocab

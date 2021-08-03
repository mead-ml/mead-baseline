import os
import string
import pytest
import random
import numpy as np
from typing import Optional, List, Set, Tuple
from itertools import chain
from eight_mile.utils import Offsets
from baseline.vectorizers import (
    Token1DVectorizer,
    Char1DVectorizer,
    Char2DVectorizer,
    TextNGramVectorizer,
    DictTextNGramVectorizer,
    BPEVectorizer1D,
    WordpieceVectorizer1D,
)

TEST_DATA = os.path.join(os.path.realpath(os.path.dirname(__file__)), "test_data")

def random_string(length: Optional[int] = None, min_: int = 3, max_: int = 6) -> str:
    length = length if length is not None else random.randint(min_, max_ - 1)
    return "".join(random.choice(string.ascii_letters) for _ in range(length))


@pytest.fixture
def vocab():
    vocab = {k: i for i, k in enumerate(Offsets.VALUES)}
    vocab['<EOW>'] = len(vocab)
    for i, k in enumerate(string.ascii_lowercase, len(vocab)): vocab[k] = i
    return vocab


def test_char_2d_shapes(vocab):
    mxlen, mxwlen = np.random.randint(1, 100, size=2)
    gold_shape = (mxlen, mxwlen)
    vect = Char2DVectorizer(mxlen=mxlen, mxwlen=mxwlen)
    res, _ = vect.run([''], vocab)
    assert res.shape == gold_shape


def test_char_2d_cuts_off_mxlen(vocab):
    mxlen = 2; mxwlen = 4
    input_ = ['a', 'b', 'c']
    vect = Char2DVectorizer(mxlen=mxlen, mxwlen=mxwlen)
    res, _ = vect.run(input_, vocab)
    assert res.shape[0] == mxlen
    for i, char in enumerate(input_[:mxlen]):
        assert res[i, 0] == vocab[char]
    values = set(res.flatten().tolist())
    for char in input_[mxlen:]:
        assert vocab[char] not in values


def test_char_2d_cuts_off_mxwlen(vocab):
    mxlen = 2; mxwlen = 4
    input_ = ['aaaabbbb', 'cccc']
    gold = np.array([[vocab['a']] * mxwlen, [vocab['c']] * mxwlen], dtype=int)
    vect = Char2DVectorizer(mxlen=mxlen, mxwlen=mxwlen)
    res, _ = vect.run(input_, vocab)
    np.testing.assert_equal(res, gold)


def test_char_2d_valid_length(vocab):
    mxlen, mxwlen = np.random.randint(3, 15, size=2)
    my_len = np.random.randint(1, mxlen)
    input_ = ['a'] * my_len
    vect = Char2DVectorizer(mxlen=mxlen, mxwlen=mxwlen)
    _, lens = vect.run(input_, vocab)
    assert lens == my_len


def test_char_2d_valid_length_cutoff(vocab):
    mxlen, mxwlen = np.random.randint(3, 15, size=2)
    my_len = mxlen + np.random.randint(5, 10)
    input_ = ['a'] * my_len
    vect = Char2DVectorizer(mxlen=mxlen, mxwlen=mxwlen)
    _, lens = vect.run(input_, vocab)
    assert my_len > mxlen
    assert lens == mxlen


def test_char_2d_run_values(vocab):
    mxlen, mxwlen = np.random.randint(3, 15, size=2)
    input_ = [chr(i + 97) * mxwlen for i in range(mxlen)]
    vect = Char2DVectorizer(mxlen=mxlen, mxwlen=mxwlen)
    res, _ = vect.run(input_, vocab)
    for i, word in enumerate(input_):
        for j, char in enumerate(word):
            assert res[i, j] == vocab[char]


def test_char_1d_shape(vocab):
    mxlen = np.random.randint(3, 15)
    input_ = ['a']
    vect = Char1DVectorizer(mxlen=mxlen)
    res, _ = vect.run(input_, vocab)
    assert res.shape == (mxlen,)


def test_char_1d_cut_off_mxlen(vocab):
    mxlen = np.random.randint(3, 15)
    extra = np.random.randint(5, 10)
    input_ = ['a' * mxlen + 'b' * extra]
    vect = Char1DVectorizer(mxlen=mxlen)
    res, _ = vect.run(input_, vocab)
    assert res.shape == (mxlen,)
    assert all(res == vocab['a'])
    assert vocab['b'] not in res


def test_char_1d_no_eow(vocab):
    del vocab['<EOW>']
    mxlen = np.random.randint(3, 15)
    input_ = ['a']
    vect = Char1DVectorizer(mxlen=mxlen)
    res, _ = vect.run(input_, vocab)
    assert res[0] == vocab['a']
    assert res[1] == Offsets.PAD


def test_text_ngrams():
    tokens = ["The", "brown", "dog", "walked", "past", "the", "white", "spotted", "one", "."]

    n = 3
    l = 4

    v = TextNGramVectorizer(filtsz=n)
    v.mxlen = l
    cnt = v.count(['<PAD>'] + tokens + ['<PAD>'])
    vocab = {}
    for i, word in enumerate(cnt.keys()):
        vocab[word] = i

    a, length = v.run(tokens, vocab)
    assert np.allclose(a, np.arange(0, 4))
    assert length == l

    tokens = [{"text": t} for t in tokens]

    v = DictTextNGramVectorizer(filtsz=n)
    v.mxlen = 100

    a, length = v.run(tokens, vocab)
    assert np.allclose(a[:length], np.arange(0, len(tokens)))


def test_default_label_indices():
    num_tokens = random.randint(1, 100)
    tokens = [random_string() for _ in range(num_tokens)]
    vec = Token1DVectorizer()
    assert vec.valid_label_indices(tokens) == [i for i in range(num_tokens)]


def test_default_label_indices_generator():
    num_tokens = random.randint(1, 100)
    tokens = (random_string() for _ in range(num_tokens))
    vec = Token1DVectorizer()
    assert vec.valid_label_indices(tokens) == [i for i in range(num_tokens)]


def bpe_tokens(
    tokens: List[str],
    break_p: float = 0.4,
    specials: Set[str] = None,
    sentinal: str = "@@"
) -> Tuple[List[str], List[int]]:
    specials = set() if specials is None else specials
    indices = []
    new_tokens = []
    i = 0
    for token in tokens:
        if token in specials:
            i += 1
            new_tokens.append(token)
            continue
        indices.append(i)
        subword = []
        for c in token:
            if random.random() > break_p:
                subword.append(c)
            else:
                if subword:
                    new_tokens.append("".join(chain(subword, [sentinal])))
                    i += 1
                subword = [c]
        if subword:
            new_tokens.append("".join(subword))
        i += 1
    return new_tokens, indices


def break_wp(word: str, break_p: float = 0.4, sentinel: str = "##") -> List[str]:
    subwords = []
    subword = []
    for c in word:
        if random.random() > break_p:
            subword.append(c)
        else:
            if subword:
                subwords.append("".join(subword))
            subword = [c]
    if subword:
        subwords.append("".join(subword))
    subwords = [s if i == 0 else sentinel + s for i, s in enumerate(subwords)]
    return subwords


def wp_tokens(
    tokens: List[str],
    break_p: float=0.4,
    specials: Set[str] = None,
    sentinel: str = "##"
) -> Tuple[List[str], List[int]]:
    specials = set() if specials is None else specials
    indices = []
    new_tokens = []
    i = 0
    prev = False
    for token in tokens:
        if token in specials:
            i += 1
            new_tokens.append(token)
            continue
        indices.append(i)
        subwords = break_wp(token, break_p, sentinel)
        print(subwords)
        new_tokens.extend(subwords)
        i += len(subwords)
    return new_tokens, indices


def add_specials(tokens: List[str], specials: Set[str], start_prob: float = 0.5, insert_prob: float = 0.2) -> List[str]:
    specials: List[str] = list(specials)
    if random.random() < 0.5:
        tokens.insert(0, specials[0])
    i = 1
    while i < len(tokens):
        if random.random() < 0.2:
            tokens.insert(i, random.choice(specials))
            i += 1
        i += 1
    return tokens


def test_bpe_label_indices():
    pytest.importorskip("fastBPE")
    num_tokens = random.randint(1, 100)
    tokens = [random_string() for _ in range(num_tokens)]
    bpe = BPEVectorizer1D(model_file=os.path.join(TEST_DATA, "codes.30k"), vocab_file=os.path.join(TEST_DATA, "vocab.30k"))
    tokens = add_specials(tokens, bpe.special_tokens)
    bpe_toks, gold_indices = bpe_tokens(tokens, specials=bpe.special_tokens)
    indices = bpe.valid_label_indices(bpe_toks)
    assert len(indices) == num_tokens
    assert indices == gold_indices


def test_bpe_label_indices_generator():
    pytest.importorskip("fastBPE")
    num_tokens = random.randint(1, 100)
    tokens = [random_string() for _ in range(num_tokens)]
    bpe = BPEVectorizer1D(model_file=os.path.join(TEST_DATA, "codes.30k"), vocab_file=os.path.join(TEST_DATA, "vocab.30k"))
    tokens = add_specials(tokens, bpe.special_tokens)
    bpe_toks, gold_indices = bpe_tokens(tokens, specials=bpe.special_tokens)
    indices = bpe.valid_label_indices((t for t in bpe_toks))
    assert len(indices) == num_tokens
    assert indices == gold_indices


def test_wp_label_indices():
    num_tokens = random.randint(1, 10)
    tokens = [random_string() for _ in range(num_tokens)]
    wp = WordpieceVectorizer1D(vocab_file=os.path.join(TEST_DATA, "bert-base-uncased-vocab.txt"))
    tokens = add_specials(tokens, wp.special_tokens)
    wp_toks, gold_indices = wp_tokens(tokens, specials=wp.special_tokens, sentinel=wp.subword_sentinel)
    indices = wp.valid_label_indices(wp_toks)
    assert len(indices) == num_tokens
    assert indices == gold_indices


def test_wp_label_indices_generator():
    num_tokens = random.randint(1, 10)
    tokens = [random_string() for _ in range(num_tokens)]
    wp = WordpieceVectorizer1D(vocab_file=os.path.join(TEST_DATA, "bert-base-uncased-vocab.txt"))
    tokens = add_specials(tokens, wp.special_tokens)
    wp_toks, gold_indices = wp_tokens(tokens, specials=wp.special_tokens, sentinel=wp.subword_sentinel)
    indices = wp.valid_label_indices((t for t in wp_toks))
    assert len(indices) == num_tokens
    assert indices == gold_indices

import string
import pytest
import numpy as np
from baseline.utils import Offsets
from baseline.vectorizers import Char2DVectorizer, Char1DVectorizer


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

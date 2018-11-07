import random
from copy import deepcopy
import pytest
from mock import MagicMock, patch
from baseline.reader import _filter_vocab, num_lines


@pytest.fixture
def data():
    keys = ['1', '2']
    words = ['a', 'b', 'c']
    vocab = {
        '1': {
            'a': 2,
            'b': 4,
            'c': 8,
        },
        '2': {
            'a': 2,
            'b': 4,
            'c': 8,
        },
    }
    return keys, words, vocab


def test_filters_one(data):
    keys, words, vocab = data
    gold = deepcopy(vocab)
    vectorizers = {k: MagicMock() for k in keys}
    vectorizers[keys[0]].min_f = 5
    vectorizers[keys[1]].min_f = 2
    vocab = _filter_vocab(vocab, vectorizers)
    assert 'a' not in vocab[keys[0]]
    assert 'b' not in vocab[keys[0]]
    assert 'c' in vocab[keys[0]]
    assert vocab[keys[1]] == gold[keys[1]]


def test_filters_both(data):
    keys, words, vocab = data
    vectorizers = {k: MagicMock() for k in keys}
    vectorizers[keys[0]].min_f = 6
    vectorizers[keys[1]].min_f = 6
    vocab = _filter_vocab(vocab, vectorizers)
    assert 'a' not in vocab[keys[0]] and 'a' not in vocab[keys[1]]
    assert 'b' not in vocab[keys[0]] and 'b' not in vocab[keys[1]]
    assert 'c' in vocab[keys[0]] and 'c' in vocab[keys[1]]


def test_no_filters(data):
    keys, words, vocab = data
    gold = deepcopy(vocab)
    vectorizers = {k: MagicMock() for k in keys}
    vectorizers[keys[0]].min_f = -1
    vectorizers[keys[1]].min_f = -1
    vocab = _filter_vocab(vocab, vectorizers)
    assert vocab == gold


def test_num_lines():
    gold = random.randint(0, 100)
    file_name = "replace with a random name"
    with patch('baseline.reader.codecs.open') as open_patch:
        file_mock = MagicMock()
        iter_mock = MagicMock()
        file_mock.__enter__.return_value = iter_mock
        iter_mock.__iter__.return_value = range(gold)
        open_patch.return_value = file_mock
        lines = num_lines(file_name)
        assert lines == gold
        open_patch.assert_called_once_with(file_name, encoding='utf-8', mode='r')


def test_num_lines_closes_file():
    gold = random.randint(0, 100)
    file_name = "replace with a random name"
    with patch('baseline.reader.codecs.open') as open_patch:
        file_mock = MagicMock()
        iter_mock = MagicMock()
        file_mock.__enter__.return_value = iter_mock
        iter_mock.__iter__.return_value = range(gold)
        open_patch.return_value = file_mock
        lines = num_lines(file_name)
        file_mock.__exit__.assert_called_once()

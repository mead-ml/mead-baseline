import os
import random
from copy import deepcopy
from collections import Counter
import pytest
import numpy as np
from mock import MagicMock, patch, call
from baseline.reader import (
    _filter_vocab,
    num_lines,
    _read_from_col,
    _build_vocab_for_col,
    _check_lens,
    TSVSeqLabelReader,
    TSVParallelCorpusReader,
    MultiFileParallelCorpusReader,
)


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


TEST_LOC = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'test_data')


def test_filters_one(data):
    keys, words, vocab = data
    gold = deepcopy(vocab)
    min_f = {keys[0]: 5, keys[1]: 2}
    vocab = _filter_vocab(vocab, min_f)
    assert 'a' not in vocab[keys[0]]
    assert 'b' not in vocab[keys[0]]
    assert 'c' in vocab[keys[0]]
    assert vocab[keys[1]] == gold[keys[1]]


def test_filters_both(data):
    keys, words, vocab = data
    min_f = dict.fromkeys(keys, 6)
    vocab = _filter_vocab(vocab, min_f)
    assert 'a' not in vocab[keys[0]] and 'a' not in vocab[keys[1]]
    assert 'b' not in vocab[keys[0]] and 'b' not in vocab[keys[1]]
    assert 'c' in vocab[keys[0]] and 'c' in vocab[keys[1]]


def test_no_filters(data):
    keys, words, vocab = data
    min_f = dict.fromkeys(keys, -1)
    with patch('baseline.reader.filter') as filt_mock:
        filt_mock.return_value = [('a', 1)]
        _ = _filter_vocab(vocab, min_f)
    filt_mock.assert_not_called()


def test_one_filters(data):
    keys, words, vocab = data
    min_f = {keys[0]: -1, keys[1]: 1}
    with patch('baseline.reader.filter') as filt_mock:
        filt_mock.return_value = [('a', 1)]
        _ = _filter_vocab(vocab, min_f)
    filt_mock.assert_called_once()


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


def test_read_cols_single_col_file():
    gold = [['one'], ['two'], ['three'], ['four'], ['five']]
    data = _read_from_col(0, [os.path.join(TEST_LOC, 'single_col.txt')])
    assert data == gold


def test_read_cols_multi_col_file():
    gold = [['one2'], ['two2'], ['three2'], ['four2'], ['five2']]
    data = _read_from_col(1, [os.path.join(TEST_LOC, 'multi_col.txt')])
    assert data == gold


def test_read_cols_multi_word_col():
    gold = [['one', 'two'], ['three', 'four'], ['five', 'six'], ['seven', 'eight'], ['nine', 'ten']]
    data = _read_from_col(2, [os.path.join(TEST_LOC, 'multi_col.txt')])
    assert data == gold


def test_vocab_col_calls_counts():
    dummy = ['text']
    with patch('baseline.reader._read_from_col') as read_patch:
        read_patch.return_value = dummy
        vects = {'a': MagicMock(), 'b': MagicMock(), 'c': MagicMock()}
        _ = _build_vocab_for_col(0, None, vects)
        for vect in vects.values():
            vect.count.assert_called_once_with(dummy[0])


def test_vocab_col_calls_read():
    file_name = 'fake'
    col = np.random.randint(0, 5)
    with patch('baseline.reader._read_from_col') as read_patch:
        _ = _build_vocab_for_col(col, file_name, {})
        read_patch.assert_called_once_with(col, file_name, r'\t', r'\s')


def test_vocab_col_uses_text():
    file_name = 'fake'
    fake_text = ['fake', 'text']
    col = np.random.randint(0, 5)
    with patch('baseline.reader._read_from_col') as read_patch:
        _ = _build_vocab_for_col(col, file_name, {}, fake_text)
        read_patch.assert_not_called()


class VectMock(MagicMock):
    def count(self, x):
        return Counter(x)


def test_tsv_unstruct_build_vocab():
    gold_vocab = {'vect': {'a': 3, 'b': 2, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1, 'h': 1, 'i': 1}}
    gold_labels = {'1', '2', '3'}
    file_name = 'tsv_unstruct_file.tsv'
    vocab, labels = TSVSeqLabelReader({'vect': VectMock()}).build_vocab(os.path.join(TEST_LOC, file_name))
    assert vocab == gold_vocab
    assert set(labels) == gold_labels


GOLD_SOURCE = {'a': 3, 'c': 2, 'e': 1}
GOLD_TARGET = {'b': 3, 'd': 2, 'f': 1, '<GO>': 3}


def test_tsv_parallel_build_vocab():
    reader = TSVParallelCorpusReader(vectorizers={'tgt': VectMock(), 'src': VectMock()})
    src_vocab, tgt_vocab = reader.build_vocabs([os.path.join(TEST_LOC, 'tsv_parallel.tsv')])
    assert src_vocab['src'] == GOLD_SOURCE
    assert tgt_vocab == GOLD_TARGET


def test_parallel_multifile_build_vocab():
    reader = MultiFileParallelCorpusReader(vectorizers={'tgt': VectMock(), 'src': VectMock()}, pair_suffix=['1', '2'])
    files = os.path.join(TEST_LOC, 'multi_parallel')
    src_vocab, tgt_vocab = reader.build_vocabs([files])
    assert src_vocab['src'] == GOLD_SOURCE
    assert tgt_vocab == GOLD_TARGET


def test_check_for_neg_one():
    gold = set()
    vects = {}
    for i in range(np.random.randint(2, 10)):
        vect = MagicMock()
        if np.random.rand() > 0.5:
            vect.mxlen = -1
            gold.add(i)
        else:
            vect.mxlen = 100
        vects[i] = vect
    fails = _check_lens(vects)
    assert fails == gold

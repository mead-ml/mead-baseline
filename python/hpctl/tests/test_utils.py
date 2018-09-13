from copy import deepcopy
import pytest
from hpctl.utils import remove_monitoring, Label


@pytest.fixture
def data():
    return {
        'a': 1,
        'b': {
            'c': 2,
            'd': 3,
        },
        'e': 4
    }


def test_remove_first_layer(data):
    gold = deepcopy(data)
    del gold['a']

    keys = {('a',)}
    res = remove_monitoring(data, keys)
    assert res == gold

def test_remove_multiple_first(data):
    gold = {'a': 1}

    keys = {('b',), ('e',)}
    res = remove_monitoring(data, keys)
    assert res == gold

def test_remove_nested_key(data):
    gold = deepcopy(data)
    del gold['b']['c']

    keys = {('b', 'c')}
    res = remove_monitoring(data, keys)
    assert res == gold

def test_remove_multiple_nested(data):
    gold = deepcopy(data)
    gold['b'] = {}

    keys = {('b', 'c'), ('b', 'd')}
    res = remove_monitoring(data, keys)
    assert res == gold

def test_remove_mixed_keys(data):
    gold = deepcopy(data)
    del gold['b']['c']
    del gold['a']

    keys = {('b', 'c'), ('a',)}
    res = remove_monitoring(data, keys)
    assert res == gold

def test_remove_single_missing(data):
    gold = deepcopy(data)

    keys = {('x',)}
    res = remove_monitoring(data, keys)
    assert res == gold

def test_remove_nested_missing_first(data):
    gold = deepcopy(data)

    keys = {('x', 'c')}
    res = remove_monitoring(data, keys)
    assert res == gold


def test_remove_nested_missing_last(data):
    gold = deepcopy(data)

    keys = {('b', 'x')}
    res = remove_monitoring(data, keys)
    assert res == gold


def test_remove_nested_missing_both(data):
    gold = deepcopy(data)

    keys = {('y', 'x')}
    res = remove_monitoring(data, keys)
    assert res == gold


def test_remove_one_missing_one_good(data):
    gold = deepcopy(data)
    del gold['a']

    keys = {('x', 'y'), ('a',)}
    res = remove_monitoring(data, keys)
    assert res == gold


def test_label_equal(data):
    l1 = Label('123', 'abc')
    l2 = Label('123', 'abc')
    assert not l1 is l2
    assert l1 == l2


def test_label_not_equal(data):
    l1 = Label('123', 'abc')
    l2 = Label('123', 'def')
    assert not l1 is l2
    assert l1 != l2


def test_label_same_hash(data):
    l1 = Label('123', 'abc')
    l2 = Label('123', 'abc')
    assert not l1 is l2
    assert hash(l1) == hash(l2)


def test_label_share_dict_keys(data):
    l1 = Label('123', 'abc')
    l2 = Label('123', 'abc')
    d = {l1: None}
    assert l2 in d

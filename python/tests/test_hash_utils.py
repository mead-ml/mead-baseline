import random
from copy import deepcopy
import mock
import pytest
from mead.utils import remove_extra_keys, order_json, hash_config


@pytest.fixture
def data():
    return {
        'a': 1,
        'b': {
            'c': 2,
            'd': 3,
        },
        'e': 4,
    }


@pytest.fixture
def mixed():
    return {
        'e': 4,
        'b': {
            'd': 3,
            'c': 2,
        },
        'a': 1,
    }


def test_hash_config_fixes_json(data):
    with mock.patch('mead.utils.remove_extra_keys') as strip_mock:
        strip_mock.return_value = data
        with mock.patch('mead.utils.order_json') as order_mock:
           order_mock.return_value = data
           hash_config(data)
    strip_mock.assert_called_once_with(data)
    order_mock.assert_called_once_with(data)


def test_order_json_e2e(data, mixed):
    res = order_json(mixed)
    assert res == data


def test_order_json_sorts_list():
    l = list(range(10))
    random.shuffle(l)
    data = {'a': l}
    gold_data = {'a': list(range(10))}
    res = order_json(data)
    assert res == gold_data


def test_remove_first_layer(data):
    gold = deepcopy(data)
    del gold['a']
    keys = {('a',)}
    res = remove_extra_keys(data, keys)
    assert res == gold


def test_remove_multiple_first(data):
    gold = {'a': 1}
    keys = {('b',), ('e',)}
    res = remove_extra_keys(data, keys)
    assert res == gold


def test_remove_nested_key(data):
    gold = deepcopy(data)
    del gold['b']['c']
    keys = {('b', 'c')}
    res = remove_extra_keys(data, keys)
    assert res == gold


def test_remove_multiple_nested(data):
    gold = deepcopy(data)
    gold['b'] = {}
    keys = {('b', 'c'), ('b', 'd')}
    res = remove_extra_keys(data, keys)
    assert res == gold


def test_remove_mixed_keys(data):
    gold = deepcopy(data)
    del gold['b']['c']
    del gold['a']
    keys = {('b', 'c'), ('a',)}
    res = remove_extra_keys(data, keys)
    assert res == gold


def test_remove_single_missing(data):
    gold = deepcopy(data)
    keys = {('x',)}
    res = remove_extra_keys(data, keys)
    assert res == gold


def test_remove_nested_missing_first(data):
    gold = deepcopy(data)
    keys = {('x', 'c')}
    res = remove_extra_keys(data, keys)
    assert res == gold


def test_remove_nested_missing_last(data):
    gold = deepcopy(data)
    keys = {('b', 'x')}
    res = remove_extra_keys(data, keys)
    assert res == gold


def test_remove_nested_missing_both(data):
    gold = deepcopy(data)
    keys = {('y', 'x')}
    res = remove_extra_keys(data, keys)
    assert res == gold


def test_remove_one_missing_one_good(data):
    gold = deepcopy(data)
    del gold['a']
    keys = {('x', 'y'), ('a',)}
    res = remove_extra_keys(data, keys)
    assert res == gold

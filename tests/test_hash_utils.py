import random
from copy import deepcopy
import mock
import pytest
from mead.utils import remove_extra_keys, order_json, hash_config, sort_list_keys


@pytest.fixture
def data():
    return {
        'a': 1,
        'b': {
            'c': 2,
            'd': 3,
        },
        'e': 4,
        'f': [
            1,
            2,
            {'g': 4, 'f': 5},
            6
        ]
    }


@pytest.fixture
def mixed():
    return {
        'e': 4,
        'b': {
            'd': 3,
            'c': 2,
        },
        'f': [
            1,
            2,
            {'f': 5, 'g': 4},
            6
        ],
        'a': 1,
    }


def test_hash_config_fixes_json(data):
    with mock.patch('mead.utils.remove_extra_keys') as strip_mock:
        strip_mock.return_value = data
        with mock.patch('mead.utils.order_json') as order_mock:
           order_mock.return_value = data
           with mock.patch('mead.utils.sort_list_keys') as sort_mock:
               sort_mock.return_value = data
               hash_config(data)
    strip_mock.assert_called_once_with(data)
    order_mock.assert_called_once_with(data)
    sort_mock.assert_called_once_with(data)


def test_order_json_e2e(data, mixed):
    res = order_json(mixed)
    assert res == data


def test_order_json_doesnt_sorts_list():
    gold = list(range(10))
    random.shuffle(gold)
    data = {'a': gold}
    res = order_json(data)
    assert res['a'] == gold


def test_sort_list():
    gold = list(range(10))
    mixed = random.sample(gold, len(gold))
    data = {'a': mixed, 'b': mixed}
    res = sort_list_keys(data, keys=(('a',),))
    assert res['a'] == gold
    assert res['b'] == mixed

def shuffle_dict(data):
    shuffled = deepcopy(data)
    while str(shuffled) == str(data):
        shuffled = {k: data[k] for k in random.sample(data.keys(), len(data.keys()))}
    return shuffled

def test_sorts_subdict():
    gold = [random.random() in range(random.randint(5, 10))]
    gold_dict = {random.randint(0, 100): random.random() for _ in range(random.randint(10, 20))}
    index = random.randint(0, len(gold) - 1)
    gold[index] = gold_dict

    mixed = deepcopy(gold)
    mixed[index] = shuffle_dict(mixed[index])
    data = {'a': mixed}
    res = order_json(data)
    assert res['a'] == gold


def test_remove_first_layer(data):
    gold = deepcopy(data)
    del gold['a']
    keys = {('a',)}
    res = remove_extra_keys(data, keys)
    assert res == gold


def test_remove_multiple_first(data):
    gold = {'a': 1}
    keys = {('b',), ('e',), ('f',)}
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

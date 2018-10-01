import random
from copy import deepcopy
import pytest
from hpctl.utils import Label


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


def test_label_equal(data):
    l1 = Label('123', '456', 'abc')
    l2 = Label('123', '456', 'abc')
    assert not l1 is l2
    assert l1 == l2


def test_label_not_equal(data):
    l1 = Label('123', '456', 'abc')
    l2 = Label('123', '456', 'def')
    assert not l1 is l2
    assert l1 != l2


def test_label_same_hash(data):
    l1 = Label('123', '456', 'abc')
    l2 = Label('123', '456', 'abc')
    assert not l1 is l2
    assert hash(l1) == hash(l2)


def test_label_share_dict_keys(data):
    l1 = Label('123', '456', 'abc')
    l2 = Label('123', '456', 'abc')
    d = {l1: None}
    assert l2 in d


def test_label_unpacking():
    exp = str(random.randint(1000, 9000))
    sha1 = str(random.randint(1000, 9000))
    name = str(random.randint(1000, 9000))
    gold = {
        'exp': exp,
        'sha1': sha1,
        'name': name
    }
    label = Label(exp, sha1, name)
    res = dict(**label)
    assert res == gold

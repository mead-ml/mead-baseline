import pytest
import numpy as np
from mead.tasks import Task


def test_get_min_f_loader_backoff():
    name = 'a'
    gold_val = np.random.randint(5, 10)
    fake = np.random.randint(10, 100)
    c = {
        'features': [{'name': name}],
        'loader': {
            'min_f': gold_val
        },
        'preproc': {
            'min_f': fake
        },
    }
    gold = {name: gold_val}
    cutoffs = Task._get_min_f(c)
    assert cutoffs == gold


def test_get_min_f_preproc_backoff():
    name = 'a'
    gold_val = np.random.randint(5, 10)
    c = {
        'features': [{'name': name}],
        'loader': {
        },
        'preproc': {
            'min_f': gold_val
        },
    }
    gold = {name: gold_val}
    cutoffs = Task._get_min_f(c)
    assert cutoffs == gold


def test_get_min_f_no_backoff():
    name = 'a'
    default = -1
    c = {
        'features': [{'name': name}],
        'loader': {
        },
    }
    gold = {name: default}
    cutoffs = Task._get_min_f(c)
    assert cutoffs == gold


def test_get_min_f_for_each_feature():
    names = ['a', 'b']
    gold_vals = np.random.randint(5, 10, size=len(names))
    c = {
        'features': [{'name': n, 'min_f': mf} for n, mf in zip(names, gold_vals)],
        'loader': {},
    }
    gold = {n: m for n, m in zip(names, gold_vals)}
    cutoffs = Task._get_min_f(c)
    assert cutoffs == gold


def test_get_min_f_each_feat_preset():
    names = np.random.choice(np.arange(1000), replace=False, size=np.random.randint(10, 100))
    c = {
        'features': [{'name': n for n in names}],
        'loader': {},
    }
    cutoffs = Task._get_min_f(c)
    for name in names:
        assert name in names



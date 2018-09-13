import random
from collections import OrderedDict
import pytest
from mock import MagicMock, call
from hpctl.sample import Sampler, GridSampler, ConfigSampler, log_sample


def test_grid_sample_len():
    gs = GridSampler(lambda x: None)
    values = {
        "a": [1, 2, 3],
        "b": [4, 5]
    }
    gs.values = values
    gold_len = 6
    assert len(gs) == gold_len


def test_grid_sample():
    gs = GridSampler(lambda x: None)
    values = OrderedDict()
    values["a"] = [1, 2, 3]
    values["b"] = [4, 5]
    gs.values = values
    gold_values = [
        {"a": 1, "b": 4},
        {"a": 1, "b": 5},
        {"a": 2, "b": 4},
        {"a": 2, "b": 5},
        {"a": 3, "b": 4},
        {"a": 3, "b": 5},
    ]
    for gold_value in gold_values:
        assert gs.sample() == gold_value


def test_grid_sample_wraps():
    gs = GridSampler(lambda x: None)
    values = OrderedDict()
    values["a"] = [1]
    values["b"] = [4, 5]
    gs.values = values
    gold_values = [
        {"a": 1, "b": 4},
        {"a": 1, "b": 5},
    ]
    for _ in gold_values:
        gs.sample()
    assert gs.sample() == gold_values[0]


def test_grid_sampler_all_keys_in_sample():
    gs = GridSampler(lambda x: None)
    values = {random.randint(0, 5): [1], random.randint(5, 10): [2]}
    gs.values = values
    sampled = gs.sample()
    for k in values:
        assert k in sampled


def test_log_sample_range():
    min_ = random.randint(1, 5)
    max_ = random.randint(5, 10)

    def real_test(min_, max_):
        assert min_ <= log_sample(min_, max_).item() <= max_

    for _ in range(100):
        real_test(min_, max_)


def test_sampler_has_all_keys():
    s = Sampler("test", lambda x: None, MagicMock)
    values = {random.randint(0, 5): [1], random.randint(5, 10): [2]}
    s.values = values
    sampled = s.sample()
    for k in values:
        assert k in sampled


def test_config_sampler_find():
    sample = MagicMock()
    data = {
        'a': {
            'type': 1,
            'a': 1
        },
        'b': {
            'c': {
                'type': 1,
                'a': 2
            }
        },
        'd': {
            'type': 2
        }
    }
    ConfigSampler._find(data, 1, sample)
    assert call({}, 'c', data['b']['c']) in sample.call_args_list
    assert call({}, 'a', data['a']) in sample.call_args_list
    assert ({}, 2, data['d']) not in sample.call_args_list


def test_keys_are_tuples():
    def sample(d, k, v):
        d[(k,)] = None

    data = {
        'a': {
            'type': 1,
            'a': 1
        },
        'b': {
            'c': {
                'type': 1,
                'a': 2
            }
        },
        'd': {
            'type': 2
        }
    }
    found = ConfigSampler._find(data, 1, sample)
    assert ('a',) in found
    assert ('b', 'c') in found

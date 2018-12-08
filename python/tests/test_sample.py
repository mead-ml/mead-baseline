import pytest
import numpy as np
from baseline.utils import topk

@pytest.fixture
def setup():
    input_ = np.random.rand(100)
    k = np.random.randint(2, 20)
    top = np.random.choice(np.arange(len(input_)), size=k, replace=False)
    add = np.max(input_) + np.arange(1, len(top) + 1)
    input_[top] = add
    return input_, top, k


def test_k_drawn(setup):
    input_, top, k = setup
    result = topk(k, input_)
    assert len(result) == k


def test_k_are_correct(setup):
    input_, top, k = setup
    result = topk(k, input_)
    for x in top:
        assert x in result


def test_k_in_order(setup):
    input_, top, k = setup
    result = topk(k, input_)
    start = -1e4
    for x in top:
        assert result[x] > start
        start = result[x]


def test_k_values_are_correct(setup):
    input_, top, k = setup
    result = topk(k, input_)
    for k, v in result.items():
        assert v == input_[k]

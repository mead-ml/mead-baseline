import pytest
import numpy as np
from backend import LocalGPUBackend


def test_request_gpus_none_open():
    be = LocalGPUBackend()
    be.gpus_to_job = {'0': 'taken', '1': 'taken'}
    gpus = be._request_gpus(1)
    assert gpus is None


def test_request_gpus_one():
    gold = ['1']
    be = LocalGPUBackend()
    be.gpus_to_job = {'0': 'taken', '1': None}
    gpus = be._request_gpus(1)
    assert gpus == gold


def test_request_gpus_one_multi_open():
    gold = [['1'], ['0']]
    be = LocalGPUBackend()
    be.gpus_to_job = {'0': None, '1': None}
    gpus = be._request_gpus(1)
    assert gpus in gold


def test_request_gpus_multi():
    gold = ['0', '1']
    be = LocalGPUBackend()
    be.gpus_to_job = {'0': None, '1': None}
    gpus = be._request_gpus(2)
    assert gpus == gold


def test_request_gpus_multi_extra_open():
    gold = {'0', '1', '3'}
    be = LocalGPUBackend()
    be.gpus_to_job = {'0': None, '1': None, '2': 'taken', '3': None}
    gpus = be._request_gpus(2)
    for gpu in gpus:
        assert gpu in gold


def test_request_gpus_not_enough():
    be = LocalGPUBackend()
    be.gpus_to_job = {'0': None, '1': 'taken', '2': 'taken'}
    gpus = be._request_gpus(2)
    assert gpus is None


def test_reserve_gpus():
    be = LocalGPUBackend()
    num_gpus = np.random.randint(5, 10)
    be.gpus_to_job = {i: None for i in range(num_gpus)}
    gpus = list(map(str, np.random.choice(np.arange(num_gpus), replace=False, size=np.random.randint(1, num_gpus -1 ))))
    job = 'job'
    be._reserve_gpus(gpus, job)
    for gpu in gpus:
        assert be.gpus_to_job[gpu] == job
    for gpu, value in be.gpus_to_job.items():
        if gpu not in gpus:
            assert value is None

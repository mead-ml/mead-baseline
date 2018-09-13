from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pytest
from hpctl.backend import handle_gpus


@pytest.fixture
def cuda_visible():
    os.environ['CUDA_VISIBLE_DEVICES'] = "5,6"
    yield None
    del os.environ['CUDA_VISIBLE_DEVICES']


@pytest.fixture
def nv_gpu():
    os.environ['NV_GPU'] = "2,4"
    yield None
    del os.environ['NV_GPU']

@pytest.fixture
def remove_env():
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    return

def test_gpu_is_none(remove_env):
    gold_gpus = [["0"]]
    g, _ = handle_gpus(None, 1, None)
    assert g == gold_gpus


def test_gpu_is_none_env(cuda_visible):
    gold_gpus = [['5'], ['6']]
    g, _ = handle_gpus(None, 1, None)
    assert g == gold_gpus


def test_gpu_is_none_env_2(nv_gpu):
    gold_gpus = [['2'], ['4']]
    g, _ = handle_gpus(None, 1, None)
    assert g == gold_gpus


def test_num_gpus_larger():
    gpus = ['1', '3', '4']
    with pytest.raises(RuntimeError):
        handle_gpus(gpus, 12, None)


def test_gpus_is_none():
    gpus = ['0']
    gold_gpus = 1
    _, g = handle_gpus(gpus, None, None)
    assert g == gold_gpus


def test_gpus_match():
    gold_gpus = [['1', '2'], ['3', '4']]
    real_gpus = ['1', '2', '3', '4']
    gpus = 2
    rg, g = handle_gpus(real_gpus, gpus, None)
    assert g == gpus
    assert rg == gold_gpus


def test_multi_gpu_one_gpu():
    gold_gpus = [['1'], ['2'], ['4'], ['7']]
    gpus = ['1', '2', '4', '7']
    gpu = 1
    g, _ = handle_gpus(gpus, gpu, None)
    assert g == gold_gpus


def test_multi_gpus_doesnt_match():
    gold_gpus = [['1', '2', '3']]
    gpus = ['1', '2', '3', '4']
    gpu = 3
    g, _ = handle_gpus(gpus, gpu, None)
    assert g == gold_gpus


def test_parallel_limit():
    gold_gpus = [['1'], ['2'], ['3']]
    gpus = ['1', '2', '3', '4']
    gpu = 1
    parallel_limit = 3
    g_full, _ = handle_gpus(gpus, gpu, None)
    g, _ = handle_gpus(gpus, gpu, parallel_limit)
    assert len(g) == parallel_limit
    assert len(g_full) > len(g)
    assert g == gold_gpus

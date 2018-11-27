import os
from copy import deepcopy
from functools import partial
import pytest
from baseline.utils import get_env_gpus, Offsets


@pytest.fixture
def cuda_visible():
    gpus = ['2', '4']
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
    yield gpus
    del os.environ['CUDA_VISIBLE_DEVICES']


@pytest.fixture
def nv_gpu():
    gpus = ['5', '6']
    os.environ['NV_GPU'] = ','.join(gpus)
    yield gpus
    del os.environ['NV_GPU']


@pytest.fixture
def remove_envs():
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    os.environ.pop('NV_GPU', None)


def test_visible(cuda_visible):
    gpus = get_env_gpus()
    assert gpus == cuda_visible


#def test_nv_gpu(nv_gpu):
#    gpus = get_env_gpus()
#    assert gpus == nv_gpu


def test_visible_first(cuda_visible, nv_gpu):
    gpus = get_env_gpus()
    assert gpus != nv_gpu
    assert gpus == cuda_visible


def test_none(remove_envs):
    gold = ['0']
    gpus = get_env_gpus()
    assert gpus == gold


def test_offsets_double_star():
    gold = {k: i for i, k in enumerate(Offsets.VALUES)}
    assert dict(**Offsets) == gold


def test_offsets_single_star():
    gold = deepcopy(Offsets.VALUES)
    def collect(*args):
        return list(args)
    assert collect(*Offsets) == gold


def test_offsets_lengths():
    gold = len(Offsets.VALUES)
    assert len(Offsets) == gold


def test_offsets_get_item():
    for i, item in enumerate(Offsets.VALUES):
        res = Offsets[item]
        assert res == i

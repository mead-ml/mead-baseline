import os
import string
import random
from copy import deepcopy
from itertools import chain
import pytest
import numpy as np
from mock import patch
from eight_mile.utils import get_env_gpus, idempotent_append, parse_module_as_path, to_numpy, get_version, remove_extensions


@pytest.fixture
def cuda_visible():
    gpus = ["2", "4"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    yield gpus
    del os.environ["CUDA_VISIBLE_DEVICES"]


@pytest.fixture
def nv_gpu():
    gpus = ["5", "6"]
    os.environ["NV_GPU"] = ",".join(gpus)
    yield gpus
    del os.environ["NV_GPU"]


@pytest.fixture
def remove_envs():
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.pop("NV_GPU", None)


def test_visible(cuda_visible):
    gpus = get_env_gpus()
    assert gpus == cuda_visible


# def test_nv_gpu(nv_gpu):
#    gpus = get_env_gpus()
#    assert gpus == nv_gpu


def test_visible_first(cuda_visible, nv_gpu):
    gpus = get_env_gpus()
    assert gpus != nv_gpu
    assert gpus == cuda_visible


# def test_none(remove_envs):
#     gold = ['0']
#     gpus = get_env_gpus()
#     assert gpus == gold


def test_idempotent_add_missing():
    element = random.randint(0, 5)
    data = set(random.randint(0, 10) for _ in range(random.randint(5, 10)))
    data.discard(element)
    data = list(data)
    assert element not in data
    idempotent_append(element, data)
    assert element in data
    assert data[-1] == element


def test_idempotent_add_there():
    element = random.randint(0, 5)
    data = set(random.randint(0, 10) for _ in range(random.randint(5, 10)))
    data.add(element)
    data = list(data)
    random.shuffle(data)
    # Make sure element isn't at the end in this example
    data.append(None)
    assert element in data
    element_idx = data.index(element)
    data_len = len(data)
    idempotent_append(element, data)
    assert element in data
    assert data.index(element) == element_idx
    assert data[-1] != element
    assert len(data) == data_len


def test_idempotent_add_last():
    # Because the last tests forces not to be last this tests always forces last
    element = random.randint(0, 5)
    data = set(random.randint(0, 10) for _ in range(random.randint(5, 10)))
    data.discard(element)
    data = list(data)
    data.append(element)
    assert element in data
    element_idx = data.index(element)
    data_len = len(data)
    idempotent_append(element, data)
    assert element in data
    assert data.index(element) == element_idx
    assert data[-1] == element
    assert len(data) == data_len


CHARS = list(chain(string.ascii_letters, string.digits))


def rand_str(length=None, min_=3, max_=10):
    if length is None:
        length = random.randint(min_, max_)
    return "".join([random.choice(CHARS) for _ in range(length)])


def test_parse_module_as_path_file():
    file_base = rand_str()
    file_ext = rand_str()
    file_name = "{}.{}".format(file_base, file_ext)
    n, d = parse_module_as_path(file_name)
    assert n == file_base
    assert d == ""


def test_parse_module_as_path_relative():
    file_base = rand_str()
    file_ext = rand_str()
    file_name = "{}.{}".format(file_base, file_ext)
    path = os.path.join(*[rand_str() for _ in range(random.randint(1, 5))])
    gold_path = rand_str()
    assert not os.path.isabs(path)
    module_name = os.path.join(path, file_name)
    with patch("eight_mile.utils.os.path.realpath") as real_patch:
        with patch("eight_mile.utils.os.path.expanduser") as user_patch:
            real_patch.return_value = gold_path
            user_patch.return_value = path
            n, d = parse_module_as_path(module_name)
            real_patch.assert_called_once_with(path)
    assert n == file_base
    assert d == gold_path


def test_parse_module_as_path_absolute():
    file_base = rand_str()
    file_ext = rand_str()
    file_name = "{}.{}".format(file_base, file_ext)
    path = os.path.join(*[rand_str() for _ in range(random.randint(1, 5))])
    path = "/" + path
    assert os.path.isabs(path)
    module_name = os.path.join(path, file_name)
    with patch("eight_mile.utils.os.path.realpath") as real_patch:
        with patch("eight_mile.utils.os.path.expanduser") as user_patch:
            real_patch.return_value = path
            user_patch.return_value = path
            n, d = parse_module_as_path(module_name)
            real_patch.assert_called_once_with(path)
    assert n == file_base
    assert d == path


def test_to_numpy_tf2():
    tf = pytest.importorskip('tensorflow')
    if get_version(tf) < 2:
        pytest.skip("TF1.X")
    gold = np.random.rand(*np.random.randint(1, 10, size=np.random.randint(1, 5)))
    tensor = tf.convert_to_tensor(gold)
    np_ = to_numpy(tensor)
    np.testing.assert_allclose(np_, gold)


def test_to_numpy_pyt_gpu():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip("No GPU Found")
    gold = np.random.rand(*np.random.randint(1, 10, size=np.random.randint(1, 5)))
    tensor = torch.from_numpy(gold).cuda()
    np_ = to_numpy(tensor)
    np.testing.assert_allclose(np_, gold)

def test_to_numpy_pyt_detch():
    torch = pytest.importorskip('torch')
    gold = np.random.rand(*np.random.randint(1, 10, size=np.random.randint(1, 5)))
    tensor = torch.from_numpy(gold)
    tensor.requires_grad = True
    np_ = to_numpy(tensor)
    np.testing.assert_allclose(np_, gold)

def test_to_numpy_pyt_detch_gpu():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip("No GPU Found")
    gold = np.random.rand(*np.random.randint(1, 10, size=np.random.randint(1, 5)))
    tensor = torch.from_numpy(gold).cuda()
    tensor.requires_grad = True
    np_ = to_numpy(tensor)
    np.testing.assert_allclose(np_, gold)

def test_to_numpy_pyt():
    torch = pytest.importorskip('torch')
    gold = np.random.rand(*np.random.randint(1, 10, size=np.random.randint(1, 5)))
    tensor = torch.from_numpy(gold)
    np_ = to_numpy(tensor)
    np.testing.assert_allclose(np_, gold)


def test_remove_ext():
    exts = {"." + rand_str() for _ in range(np.random.randint(2, 4))}
    ext_list = list(exts)
    gold = rand_str()
    path = deepcopy(gold)
    for _ in range(np.random.randint(1, 2)):
        path += random.choice(ext_list)
    res = remove_extensions(path, exts)
    assert res == gold

def test_remove_ext_not_in_middle():
    gold = "example.bio.more-stuff"
    exts = {".bio"}
    path = deepcopy(gold)
    res = remove_extensions(path, exts)
    assert res == gold

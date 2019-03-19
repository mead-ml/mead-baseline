import os
import string
import random
from itertools import chain
from collections import namedtuple
from mock import patch, call
import pytest
from mead.utils import convert_path, find_model_version, get_output_paths


CHARS = list(chain(string.ascii_letters, string.digits))


@pytest.fixture
def file_name():
    file_ = "test_file"
    with open(file_, "w"): pass
    yield file_
    os.remove(file_)


def test_no_loc():
    file_name = "test"
    gold = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "mead", file_name))
    path = convert_path(file_name)
    assert path == gold


def test_loc():
    file_name = "test"
    start = "/dev"
    gold = os.path.join(start, file_name)
    path = convert_path(file_name, start)
    assert path == gold


def test_file_exists(file_name):
    start = "/dev"
    gold = file_name
    wrong = os.path.join(start, file_name)
    path = convert_path(file_name, start)
    assert path == gold
    assert path != wrong


def rand_str(length=None, min_=3, max_=10):
    if length is None:
        length = random.randint(min_, max_)
    return ''.join([random.choice(CHARS) for _ in range(length)])


def test_find_version():
    gold = random.randint(5, 15)
    items = list(map(str, range(gold)))
    gold = str(gold)
    with patch('mead.utils._listdir') as list_patch:
        list_patch.return_value = items
        res = find_model_version(None)
    assert res == gold


def test_find_version_gappy():
    gold = random.randint(5, 15)
    items = list(map(str, random.sample(range(gold), gold // 2)))
    items.append(str(gold - 1))
    gold = str(gold)
    with patch('mead.utils._listdir') as list_patch:
        list_patch.return_value = items
        res = find_model_version(None)
    assert res == gold


def test_find_version_non_int():
    gold = random.randint(5, 15)
    items = list(map(str, random.sample(range(gold), gold // 2)))
    items.append(str(gold - 1))
    gold = str(gold)
    items.insert(random.randint(0, len(items)), rand_str())
    with patch('mead.utils._listdir') as list_patch:
        list_patch.return_value = items
        res = find_model_version(None)
    assert res == gold


def test_find_version_none():
    gold = "1"
    items = []
    with patch('mead.utils._listdir') as list_patch:
        list_patch.return_value = items
        res = find_model_version(None)
    assert res == gold


def test_find_version_non_int_only():
    gold = "1"
    items = [rand_str for _ in range(random.randint(1, 3))]
    with patch('mead.utils._listdir') as list_patch:
        list_patch = items
        res = find_model_version(None)
    assert res == gold


names = namedtuple("d", "dir base proj name version")


@pytest.fixture
def d():
    data = []
    data.append(os.path.join(*[rand_str() for _ in range(random.randint(1, 4))]))
    data.append(os.path.basename(data[-1]))
    data.append(rand_str())
    data.append(rand_str())
    data.append(str(random.randint(1, 4)))
    return names(*data)


@pytest.fixture
def m_patch():
    with patch('mead.utils.os.makedirs') as m_patch:
        yield m_patch


def test_get_output_paths_old_remote(d, m_patch):
    gc = os.path.join(d.dir, 'client', d.base, d.version)
    gs = os.path.join(d.dir, 'server', d.base, d.version)
    c, s = get_output_paths(d.dir, None, None, d.version, True)
    assert c == gc
    assert s == gs


def test_get_output_paths_old(d, m_patch):
    gc = os.path.join(d.dir, d.version)
    gs = os.path.join(d.dir, d.version)
    c, s = get_output_paths(d.dir, None, None, d.version, False)
    assert c == gc
    assert s == gs
    assert c == s


def test_get_output_paths_project_name(d, m_patch):
    g = os.path.join(d.dir, d.proj, d.name, d.version)
    c, s = get_output_paths(d.dir, d.proj, d.name, d.version, False)
    assert c == g
    assert c == s


def test_get_output_paths_project_name_remote(d, m_patch):
    gc = os.path.join(d.dir, 'client', d.proj, d.name, d.version)
    gs = os.path.join(d.dir, 'server', d.proj, d.name, d.version)
    c, s = get_output_paths(d.dir, d.proj, d.name, d.version, True)
    assert c == gc
    assert s == gs


def test_get_output_paths_project(d, m_patch):
    g = os.path.join(d.dir, d.proj, d.version)
    c, s = get_output_paths(d.dir, d.proj, None, d.version, False)
    assert c == g
    assert c == s


def test_get_output_paths_project_remote(d, m_patch):
    gc = os.path.join(d.dir, "client", d.proj, d.version)
    gs = os.path.join(d.dir, "server", d.proj, d.version)
    c, s = get_output_paths(d.dir, d.proj, None, d.version, True)
    assert c == gc
    assert s == gs


def test_get_output_paths_name(d, m_patch):
    g = os.path.join(d.dir, d.name, d.version)
    c, s = get_output_paths(d.dir, None, d.name, d.version, False)
    assert c == g
    assert c == s


def test_get_output_paths_name_remote(d, m_patch):
    gc = os.path.join(d.dir, "client", d.name, d.version)
    gs = os.path.join(d.dir, "server", d.name, d.version)
    c, s = get_output_paths(d.dir, None, d.name, d.version, True)
    assert c == gc
    assert s == gs


def test_get_output_paths_no_version(d, m_patch):
    g = os.path.join(d.dir, d.proj, d.name, d.version)
    with patch('mead.utils.find_model_version') as v_patch:
        v_patch.return_value = d.version
        c, s = get_output_paths(d.dir, d.proj, d.name, None, False)
    assert c == g
    assert c == s


def test_get_output_paths_no_version_remote(d, m_patch):
    gc = os.path.join(d.dir, "client", d.proj, d.name, d.version)
    gs = os.path.join(d.dir, "server", d.proj, d.name, d.version)
    with patch('mead.utils.find_model_version') as v_patch:
        v_patch.return_value = d.version
        c, s = get_output_paths(d.dir, d.proj, d.name, None, True)
    assert c == gc
    assert s == gs


def test_get_output_paths_make_server(d, m_patch):
    g = os.path.join(d.dir, d.proj, d.name, d.version)
    _, _ = get_output_paths(d.dir, d.proj, d.name, d.version, False, True)
    m_patch.assert_called_once_with(g)


def test_get_output_paths_no_make_server(d, m_patch):
    _, _ = get_output_paths(d.dir, d.proj, d.name, d.version, False, False)
    m_patch.assert_not_called()


def test_get_output_paths_make_server_remote(d, m_patch):
    gs = os.path.join(d.dir, "server", d.proj, d.name, d.version)
    gc = os.path.join(d.dir, "client", d.proj, d.name, d.version)
    _, _ = get_output_paths(d.dir, d.proj, d.name, d.version, True, True)
    assert m_patch.call_args_list == [call(gc), call(gs)]


def test_get_output_paths_no_make_server_remote(d, m_patch):
    gc = os.path.join(d.dir, "client", d.proj, d.name, d.version)
    _, _ = get_output_paths(d.dir, d.proj, d.name, d.version, True, False)
    m_patch.assert_called_once_with(gc)

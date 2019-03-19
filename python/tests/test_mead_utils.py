import os
import string
import random
from itertools import chain
from collections import namedtuple
from mock import patch, call
import pytest
from mead.utils import (
    convert_path,
    get_output_paths,
    get_export_params,
    find_model_version,
)


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


def test_get_export_input_override():
    project = rand_str()
    name = rand_str()
    output_dir = os.path.join(rand_str(), rand_str())
    model_version = str(random.randint(1, 5))
    config = {
        'project': rand_str(),
        'name': rand_str(),
        'output_dir': os.path.join(rand_str(), rand_str()),
        'model_version': str(random.randint(1, 5))
    }
    o, p, n, v = get_export_params(config, output_dir, project, name, model_version)
    assert o == output_dir
    assert p == project
    assert n == name
    assert v == model_version


def test_get_export_defaults():
    o, p, n, v = get_export_params({})
    assert o == './models'
    assert p is None
    assert n is None
    assert v is None


def test_get_export_config():
    config = {
        'project': rand_str(),
        'name': rand_str(),
        'output_dir': os.path.join(rand_str(), rand_str()),
        'model_version': str(random.randint(1, 5))
    }
    o, p, n, v = get_export_params(config)
    assert o == config['output_dir']
    assert p == config['project']
    assert n == config['name']
    assert v == config['model_version']


def test_get_export_output_expanded():
    output_dir = "~/example"
    gold_output_dir = os.path.expanduser(output_dir)
    o, _, _, _ = get_export_params({}, output_dir)


def choice(in_, config, key):
    c = random.randint(1, 3)
    if c == 1:
        return in_, in_
    elif c == 2:
        return None, config[key]
    else:
        del config[key]
        return None, None


def test_get_export_params():
    def test():
        in_ = d()
        c = d()
        config = {
            'output_dir': c.dir,
            'project': c.proj,
            'name': c.name,
            'model_version': c.version,
        }
        in_output, gold_output = choice(in_.dir, config, 'output_dir')
        gold_output = './models' if gold_output is None else gold_output
        in_project, gold_project = choice(in_.proj, config, 'project')
        in_name, gold_name = choice(in_.name, config, 'name')
        in_version, gold_version = choice(in_.version, config, 'model_version')
        o, p, n, v = get_export_params(config, in_output, in_project, in_name, in_version)
        assert o == gold_output
        assert p == gold_project
        assert n == gold_name
        assert v == gold_version

    for _ in range(100):
        test()

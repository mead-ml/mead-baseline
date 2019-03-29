import os
import string
import random
from itertools import chain
from collections import namedtuple
from mock import patch, call
import pytest
from baseline.utils import str2bool
from mead.utils import (
    convert_path,
    get_output_paths,
    get_export_params,
    find_model_version,
    get_dataset_from_key
)


CHARS = list(chain(string.ascii_letters, string.digits))


@pytest.fixture
def file_name():
    file_ = "test_file"
    with open(file_, "w"):
        pass
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
    return make_data()


def make_data():
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
    exporter_type = rand_str()
    return_labels = random.choice([True, False])
    is_remote = random.choice([True, False])
    config = {
        'project': rand_str(),
        'name': rand_str(),
        'output_dir': os.path.join(rand_str(), rand_str()),
        'model_version': str(random.randint(1, 5)),
        'exporter_type': rand_str(),
        'return_labels': not return_labels,
        'is_remote': not is_remote,
    }
    o, p, n, v, e, l, r = get_export_params(config, output_dir, project, name, model_version, exporter_type, return_labels, is_remote)
    assert o == output_dir
    assert p == project
    assert n == name
    assert v == model_version
    assert e == exporter_type
    assert l == return_labels
    assert r == is_remote


def test_get_export_defaults():
    o, p, n, v, e, l, r = get_export_params({})
    assert o == './models'
    assert p is None
    assert n is None
    assert v is None
    assert e == 'default'
    assert l is False
    assert r is True


def test_get_export_config():
    config = {
        'project': rand_str(),
        'name': rand_str(),
        'output_dir': os.path.join(rand_str(), rand_str()),
        'model_version': str(random.randint(1, 5)),
        'exporter_type': rand_str(),
        'return_labels': random.choice(['true', 'false']),
        'is_remote': random.choice(['true', 'false']),
    }
    o, p, n, v, e, l, r = get_export_params(config)
    assert o == config['output_dir']
    assert p == config['project']
    assert n == config['name']
    assert v == config['model_version']
    assert e == config['exporter_type']
    assert l == str2bool(config['return_labels'])
    assert r == str2bool(config['is_remote'])


def test_get_export_type_in_config():
    config = {'type': rand_str()}
    _, _, _, _, e, _, _ = get_export_params(config)
    assert e == config['type']


def test_get_export_output_expanded():
    output_dir = "~/example"
    gold_output_dir = os.path.expanduser(output_dir)
    o, _, _, _, _, _, _ = get_export_params({}, output_dir)
    assert o == gold_output_dir


def test_get_export_str2bool_called():
    return_labels = random.choice(['true', 'false'])
    is_remote = random.choice(['true', 'false'])
    with patch('mead.utils.str2bool') as b_patch:
        _ = get_export_params({}, return_labels=return_labels, is_remote=is_remote)
        assert b_patch.call_args_list == [call(return_labels), call(is_remote)]


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
        in_ = make_data()
        c = make_data()
        config = {
            'output_dir': c.dir,
            'project': c.proj,
            'name': c.name,
            'model_version': c.version,
            'exporter_type': rand_str(),
            'return_labels': random.choice(['true', 'false']),
            'is_remote': random.choice(['true', 'false']),
        }
        in_output, gold_output = choice(in_.dir, config, 'output_dir')
        gold_output = './models' if gold_output is None else gold_output
        in_project, gold_project = choice(in_.proj, config, 'project')
        in_name, gold_name = choice(in_.name, config, 'name')
        in_version, gold_version = choice(in_.version, config, 'model_version')
        in_export, gold_export = choice(rand_str(), config, 'exporter_type')
        gold_export = gold_export if gold_export is not None else 'default'
        in_labels, gold_labels = choice(random.choice(['true', 'false']), config, 'return_labels')
        gold_labels = str2bool(gold_labels) if gold_labels is not None else False
        in_remote, gold_remote = choice(random.choice(['true', 'false']), config, 'is_remote')
        gold_remote = str2bool(gold_remote) if gold_remote is not None else True
        o, p, n, v, e, l, r = get_export_params(
            config,
            in_output,
            in_project, in_name,
            in_version,
            in_export,
            in_labels,
            in_remote,
        )
        assert o == gold_output
        assert p == gold_project
        assert n == gold_name
        assert v == gold_version
        assert e == gold_export
        assert l == gold_labels
        assert r == gold_remote

    for _ in range(100):
        test()


def test_dataset_formats():
    keys = {'1': 1,
            '1:1978': 7,
            '2:1996': 2,
            '2:20190327': 3,
            '2:2019-03-28': 42
    }

    # Test exact match first
    assert get_dataset_from_key('1', keys) == 1
    # Test where not exact that we get last date
    assert get_dataset_from_key('2', keys) == 42
    # Test where we do not get last date that we get an exception
    try:
        j = get_dataset_from_key('3', keys)
        assert j is None
    except:
        pass

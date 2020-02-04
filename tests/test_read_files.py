import os
import json
import mock
import pytest
from eight_mile.utils import read_config_file, read_json, read_yaml, read_config_stream


@pytest.fixture
def gold_data():
    return {"a": 1, "b": {"c": 2}}


@pytest.fixture
def env(gold_data):
    env_name = "TEST"
    os.environ[env_name] = json.dumps(gold_data)
    yield "$" + env_name
    del os.environ[env_name]


data_loc = os.path.realpath(os.path.dirname(__file__))
data_loc = os.path.join(data_loc, "test_data")


def test_read_json(gold_data):
    data = read_json(os.path.join(data_loc, "test_json.json"))
    assert data == gold_data


def test_read_json_default_value():
    gold_default = {}
    data = read_json(os.path.join(data_loc, "not_there.json"))
    assert data == gold_default


def test_read_json_given_default():
    gold_default = "default"
    data = read_json(os.path.join(data_loc, "not_there.json"), gold_default)
    assert data == gold_default


def test_read_json_strict():
    with pytest.raises(IOError):
        read_json(os.path.join("not_there.json"), strict=True)


def test_read_yaml(gold_data):
    pytest.importorskip("yaml")
    data = read_yaml(os.path.join(data_loc, "test_yaml.yml"))
    assert data == gold_data


def test_read_yaml_default_value():
    pytest.importorskip("yaml")
    gold_default = {}
    data = read_yaml(os.path.join(data_loc, "not_there.yml"))
    assert data == gold_default


def test_read_yaml_given_default():
    pytest.importorskip("yaml")
    gold_default = "default"
    data = read_yaml(os.path.join("not_there.yml"), gold_default)
    assert data == gold_default


def test_read_yaml_strict():
    pytest.importorskip("yaml")
    with pytest.raises(IOError):
        read_yaml(os.path.join("not_there.yml"), strict=True)


def test_read_config_json_dispatch():
    file_name = "example.json"
    with mock.patch("eight_mile.utils.read_json") as read_patch:
        read_config_file(file_name)
    read_patch.assert_called_once_with(file_name, strict=True)


def test_read_config_ymal_dispatch():
    pytest.importorskip("yaml")
    file_name = "example.yml"
    with mock.patch("eight_mile.utils.read_yaml") as read_patch:
        read_config_file(file_name)
    read_patch.assert_called_once_with(file_name, strict=True)


def test_read_config_stream_file():
    file_name = os.path.join(data_loc, "test_json.json")
    with mock.patch("eight_mile.utils.read_config_file") as read_patch:
        read_config_stream(file_name)
    read_patch.assert_called_once_with(file_name)


def test_read_config_stream_env(env, gold_data):
    data = read_config_stream(env)
    assert data == gold_data


def test_read_config_stream_str(gold_data):
    input_ = json.dumps(gold_data)
    data = read_config_stream(input_)
    assert data == gold_data

import os
import pytest
from mead.utils import convert_path

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

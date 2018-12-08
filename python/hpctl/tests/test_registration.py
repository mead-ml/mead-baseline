import pytest
from hpctl.utils import register
from . import r_str


class CreateClass(object):
    @classmethod
    def create(cls):
        return cls()


class InitClass(object): pass


def test_errors_on_re_regesiter():
    R = {}
    name = r_str()
    register(CreateClass, R, name)
    with pytest.raises(Exception):
        register(InitClass, R, name)

def test_adds_create():
    R = {}
    register(CreateClass, R)
    assert R['CreateClass'] == CreateClass.create


def test_adds_init():
    R = {}
    register(InitClass, R)
    assert R['InitClass'] == InitClass


def test_uses_name():
    R = {}
    name = r_str()
    register(InitClass, R, name=name)
    assert 'InitClass' not in R
    assert name in R


def test_uses_class_name():
    R = {}
    register(InitClass, R)
    register(CreateClass, R)
    assert 'InitClass' in R
    assert 'CreateClass' in R

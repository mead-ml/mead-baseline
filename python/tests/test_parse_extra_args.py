import pytest
from mead.utils import parse_extra_args


def test_all_base_name_appear():
    base_names = ['a', 'b']
    reporting = parse_extra_args(base_names, [])
    for name in base_names:
        assert name in reporting


def test_all_special_names_grabbed():
    base_names = ['a', 'b']
    special_names = ['a:one', 'a:two']
    gold_special_names = ['one', 'two']
    reporting = parse_extra_args(base_names, special_names)
    for gold in gold_special_names:
        assert gold in reporting['a']


def test_nothing_if_no_special():
    base_names = ['a', 'b']
    special_names = ['a:one', 'a:two']
    reporting = parse_extra_args(base_names, special_names)
    assert {} == reporting['b']


def test_special_names_multiple_bases():
    base_names = ['a', 'b']
    special_names = ['a:one', 'b:two']
    reporting = parse_extra_args(base_names, special_names)
    assert 'one' in reporting['a']
    assert 'two' not in reporting['a']
    assert 'two' in reporting['b']
    assert 'one' not in reporting['b']


def test_special_shared_across_base():
    base_names = ['a', 'b']
    special_names = ['--b:one', 'b', '--a:one', 'a']
    reporting = parse_extra_args(base_names, special_names)
    for name in base_names:
        assert 'one' in reporting[name]
        assert reporting[name]['one'] == name


# This depends on `argparse` atm which should be mocked out eventually
def test_values_are_grabbed():
    base_names = ['a', 'b']
    special_names = ['--a:xxx', 'xxx', '--a:yyy', 'yyy', '--b:zzz', 'zzz']
    reporting = parse_extra_args(base_names, special_names)
    for name in base_names:
        for special, value in reporting[name].items():
            assert special == value

def test_extra_things_ignored():
    base_names = ['b']
    special_names = ['a:one']
    reporting = parse_extra_args(base_names, special_names)
    assert {} == reporting['b']


def test_no_base_names():
    base_names = []
    special_names = ["--visdom:name", "sst2"]
    reporting = parse_extra_args(base_names, special_names)
    assert {} == reporting

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import json
import pytest
from mock import patch
from hpctl.core import (
    force_xpctl,
    force_remote_backend,
    override_client_settings,
)
from hpctl.settings import (
    get_xpctl_settings,
)


def test_cache_overwritten():
    old = 'client_cache'
    input_ = {'datacache': old}
    gold = 'server_cache'
    override_client_settings(input_, gold, None)
    assert input_['datacache'] == gold


def test_cache_added():
    input_ = {}
    gold = 'server_cache'
    override_client_settings(input_, gold, None)
    assert input_['datacache'] == gold


def test_no_xpctl_removes_client():
    input_ = {'reporting': {'xpctl': {'label': 'client_stuff'}}}
    override_client_settings(input_, None, None)
    assert 'xpctl' not in input_['reporting']


def test_no_xpctl_leaves_others():
    key = 'visdom'
    value = {'env': 'example'}
    input_ = {'reporting': {'xpctl': {}, key: value}}
    override_client_settings(input_, None, None)
    assert 'xpctl' not in input_['reporting']
    assert key in input_['reporting']
    assert input_['reporting'][key] == value


def test_xpctl_overrides():
    input_ = {'reporting': {'xpctl': {'cred': 'client-cred'}}}
    gold = 'server-cred'
    override_client_settings(input_, None, gold)
    assert input_['reporting']['xpctl']['cred'] == gold


def test_xpctl_adds():
    input_ = {'reporting': {'xpctl': {}}}
    gold = 'server-cred'
    override_client_settings(input_, None, gold)
    assert input_['reporting']['xpctl']['cred'] == gold


def test_handles_no_xpctl():
    input_ = {'reporting': {}}
    gold = 'server-cred'
    override_client_settings(input_, None, gold)
    assert 'xpctl' not in input_


def test_handles_no_reporting():
    input_ = {}
    gold = 'server-cred'
    override_client_settings(input_, None, gold)
    assert 'reporting' not in input_


def test_force_xpctl_adds_reporting():
    input_ = {}
    force_xpctl(input_, None)
    assert 'reporting' in input_

def test_force_xpctl_adds_xpctl():
    input_ = {'reporting': {}}
    force_xpctl(input_, None)
    assert 'xpctl' in input_['reporting']

def test_force_xpctl_adds_both():
    input_ = {}
    force_xpctl(input_, None)
    assert 'reporting' in input_
    assert 'xpctl' in input_['reporting']

def test_force_xpctl_adds_label():
    gold = 'xpctl-label'
    input_ = {}
    force_xpctl(input_, gold)
    assert input_['reporting']['xpctl']['label'] == gold

def test_force_xpctl_skips_None():
    input_ = {}
    force_xpctl(input_, None)
    assert 'label' not in input_['reporting']['xpctl']

def test_force_xpctl_ignores_rest():
    gold = 'xpctl-label'
    key = 'visdom'
    value = {'example': 'example'}
    xp_key = 'user'
    xp_value = 'root'
    input_ = {
        'reporting': {
            key: value,
            'xpctl': {
                xp_key: xp_value,
            }
        }
    }
    force_xpctl(input_, gold)
    assert key in input_['reporting']
    assert input_['reporting'][key] == value
    assert xp_key in input_['reporting']['xpctl']
    assert input_['reporting']['xpctl'][xp_key] == xp_value


def test_force_remote_backend_overrides_type():
    be = {'type': 'local'}
    gold = 'remote'
    force_remote_backend(be)
    assert be['type'] == gold


def test_force_remote_backend_adds_type():
    be = {}
    gold = 'remote'
    force_remote_backend(be)
    assert be['type'] == gold


def test_force_remote_backend_defaults_host():
    be = {}
    gold = 'localhost'
    force_remote_backend(be)
    assert be['host'] == gold


def test_force_remote_backend_respects_host():
    gold = 'hostname'
    be = {'host': gold}
    force_remote_backend(be)
    assert be['host'] == gold


def test_force_remote_backend_defaults_port():
    be = {}
    gold = 5000
    force_remote_backend(be)
    assert be['port'] == gold


def test_force_remote_backend_respects_port():
    gold = 6006
    be = {'port': gold}
    force_remote_backend(be)
    assert be['port'] == gold


def test_xpctl_no_reporting():
    settings = {}
    xpctl = get_xpctl_settings(settings)
    assert xpctl is None


def test_xpctl_no_xpctl():
    settings = {'reporting_hooks': {'visdom': None}}
    xpctl = get_xpctl_settings(settings)
    assert xpctl is None


def test_xpctl_no_cred():
    settings = {'reporting_hooks': {'xpctl': {'label': 'name'}}}
    xpctl = get_xpctl_settings(settings)
    assert xpctl is None


def test_xpctl_file_cred():
    loc = 'xpctlcred_loc'
    settings = {'reporting_hooks': {'xpctl': {'cred': loc}}}
    with patch('hpctl.settings.read_config_file_or_json') as read_patch:
        _ = get_xpctl_settings(settings)
    read_patch.assert_called_once_with(loc)

from __future__ import absolute_import, division, print_function, unicode_literals

import os
from baseline.utils import read_config_file
from baseline.utils import export as exporter
from mead.utils import (
    hash_config,
    parse_extra_args,
    get_mead_settings,
    read_config_file_or_json,
)


def get_config(config, reporting, extra_args):
    mead_config = read_config_file_or_json(config)
    if reporting is not None:
        mead_config['reporting'] = parse_extra_args(reporting, extra_args)
    return mead_config


def get_settings(settings):
    mead_settings = get_mead_settings(settings)
    hpctl_settings = mead_settings.get('hpctl', {})
    return hpctl_settings, mead_settings


def get_logs(hpctl_settings, logging, hpctl_logging):
    mead_logs = read_config_file_or_json(logging)
    hpctl_logs = read_config_file_or_json(hpctl_logging)
    hpctl_logs['host'] = hpctl_settings.get('logging', {}).get('host', 'localhost')
    hpctl_logs['port'] = int(hpctl_settings.get('logging', {}).get('post', 6006))
    return hpctl_logs, mead_logs


def set_root(hpctl_settings, default='delete_me'):
    root = hpctl_settings.get('root', default)
    try:
        os.mkdir(root)
    except OSError:
        pass
    os.chdir(root)


def get_ends(hpctl_settings, extra_args):
    ends = parse_extra_args(['frontend', 'backend'], extra_args)
    fe = ends['frontend']
    # Merge dicts
    for key, val in hpctl_settings.get('frontend', {'type': 'console'}).items():
        if key not in fe:
            fe[key] = val

    be = ends['backend']
    # Merge dicts
    for key, val in hpctl_settings.get('backend', {'type': 'mp'}).items():
        if key not in be:
            be[key] = val
    if isinstance(be.get('real_gpus'), str):
        be['real_gpus'] = be['real_gpus'].split(",")
    return fe, be


def get_xpctl_settings(mead_settings):
    xpctl = mead_settings.get('reporting_hooks', {}).get('xpctl', {})
    if 'cred' not in xpctl:
        return None
    return read_config_file(os.path.expanduser(xpctl['cred']))

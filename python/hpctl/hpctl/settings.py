from __future__ import absolute_import, division, print_function, unicode_literals

import os
from baseline.utils import read_config_file
from baseline.utils import export as exporter
from mead.utils import get_mead_settings, parse_extra_args, hash_config


def get_configs(**kwargs):
    hpctl_config = read_config_file(kwargs['config'])
    mead_config = hpctl_config['mead']
    if isinstance(mead_config, str):
        mead_config = read_config_file(mead_config)
    if kwargs.get('reporting') is not None:
        mead_config['reporting'] = parse_extra_args(kwargs.get('reporting'), kwargs['unknown'])
    return hpctl_config, mead_config


def get_settings(**kwargs):
    mead_settings = get_mead_settings(kwargs['settings'])
    hpctl_settings = mead_settings.get('hpctl', {})
    return hpctl_settings, mead_settings


def get_logs(hpctl_settings, **kwargs):
    mead_logs = read_config_file(kwargs['logging'])
    hpctl_logs = read_config_file(kwargs['hpctl_logging'])
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


def get_ends(hpctl_config, hpctl_settings, **kwargs):
    ends = parse_extra_args(['frontend', 'backend'], kwargs['unknown'])
    if kwargs.get('frontend') is None:
        frontend = hpctl_config.get('frontend', hpctl_settings.get('frontend', {'type': 'console'}))
    else:
        frontend = {'type': kwargs['frontend']}
    # Merge dicts
    for key, val in frontend.items():
        if key not in ends['frontend']:
            ends['frontend'][key] = val

    if kwargs.get('backend') is None:
        backend = hpctl_config.get('backend', hpctl_settings.get('backend', {'type': 'mp'}))
    else:
        backend = {'type': kwargs['backend']}
    # Merge dicts
    for key, val in backend.items():
        if key not in ends['backend']:
            ends['backend'][key] = val
    frontend_config = ends['frontend']
    backend_config = ends['backend']
    if isinstance(backend_config.get('real_gpus'), str):
        backend_config['real_gpus'] = backend_config['real_gpus'].split(",")
    return frontend_config, backend_config


def get_xpctl_settings(mead_settings):
    xpctl = mead_settings.get('reporting_hooks', {}).get('xpctl', {})
    return read_config_file(xpctl['cred'])

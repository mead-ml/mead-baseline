from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob
import json
from baseline.utils import read_config_file
from baseline.utils import export as exporter
from xpctl.core import ExperimentRepo


__all__ = []
export = exporter(__all__)


def read_logs(file_name):
    logs = []
    with open(file_name) as f:
        for line in f:
            logs.append(json.loads(line))
    return logs


def dummy_print(s):
    pass


@export
class XPCTL(object):
    def __init__(self, label=None, **kwargs):
        super(XPCTL, self).__init__()
        self.name = label
        self.xpctl_config = kwargs
        self.repo = None

    def put_result(self, label):
        # Wait to create the experiment repo until after the fork
        if self.repo is None:
            try:
                self.repo = ExperimentRepo.create_repo(**self.xpctl_config)
            except Exception as e:
                return str(e)
        loc = os.path.join(label.exp, label.sha1, label.name)
        config_loc = os.path.join(loc, 'config.json')
        config = read_config_file(config_loc)
        task = config.get('task')
        log_loc = glob.glob(os.path.join(loc, 'reporting-*.log'))[0]
        logs = read_logs(log_loc)
        return str(self.repo.put_result(task, config, logs, print_fn=dummy_print, label=self.name))


@export
def get_xpctl(xpctl_config):
    if xpctl_config is None:
        return None
    if xpctl_config.pop('type', 'local') == 'remote':
        from hpctl.remote import RemoteXPCTL
        return RemoteXPCTL(**xpctl_config)
    return XPCTL(**xpctl_config)

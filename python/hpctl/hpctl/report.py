from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob
import json
import requests
from baseline.utils import read_config_file
from xpctl.core import ExperimentRepo


def read_logs(file_name):
    logs = []
    with open(file_name) as f:
        for line in f:
            logs.append(json.loads(line))
    return logs


class XPCTL(object):
    def __init__(self, **kwargs):
        super(XPCTL, self).__init__()
        self.repo = ExperimentRepo.create_repo(**kwargs)

    def put_result(self, label):
        loc = os.path.join(label.exp, label.sha1, label.name)
        config_loc = os.path.join(loc, 'config.json')
        config = read_config_file(config_loc)
        task = config.get('task')
        log_loc = glob.glob(os.path.join(loc, 'reporting-*.log'))[0]
        logs = read_logs(log_loc)
        self.repo.put_result(task, config, logs)


class RemoteXPCTL(object):
    def __init__(self, url, port, **kwargs):
        self.url = "http://{url}:{port}/hpctl/v1".format(url=url, port=port)

    def put_result(self, label):
        r = requests.post(
            '{url}/xpctl/putresult/{exp}/{sha1}/{name}'.format(
                url=self.url, **label)
        )

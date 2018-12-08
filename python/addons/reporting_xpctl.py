from __future__ import print_function

import os
import getpass
import socket
from baseline.reporting import EpochReportingHook
from mead.utils import read_config_file_or_json
from xpctl.core import ExperimentRepo
from baseline.reporting import register_reporting

@register_reporting(name='xpctl')
class XPCtlReporting(EpochReportingHook):
    def __init__(self, **kwargs):
        super(XPCtlReporting, self).__init__(**kwargs)
        # throw exception if the next three can't be read from kwargs
        self.cred = read_config_file_or_json(kwargs['cred'])
        self.label = kwargs.get('label', None)
        self.exp_config = read_config_file_or_json(kwargs['config_file'])
        self.task = kwargs['task']
        self.print_fn = print
        self.username = kwargs.get('user', getpass.getuser())
        self.hostname = kwargs.get('host', socket.gethostname())
        self.checkpoint_base = None
        self.checkpoint_store = kwargs.get('checkpoint_store', '/data/model-checkpoints')
        self.save_model = kwargs.get('save_model', False) # optionally save the model

        self.repo = ExperimentRepo().create_repo(**self.cred)
        self.log = []

    def _step(self, metrics, tick, phase, tick_type, **kwargs):
        """Write intermediate results to a logging memory object that ll be pushed to the xpctl repo

        :param metrics: A map of metrics to scores
        :param tick: The time (resolution defined by `tick_type`)
        :param phase: The phase of training (`Train`, `Valid`, `Test`)
        :param tick_type: The resolution of tick (`STEP`, `EPOCH`)
        :return:
        """
        msg = {'tick_type': tick_type, 'tick': tick, 'phase': phase}
        for k, v in metrics.items():
            msg[k] = v
        self.log.append(msg)

    def done(self):
        """Write the log to the xpctl database"""
        if self.save_model:
           self.backend = self.exp_config.get('backend', 'default')
           backends = {'default': 'tf', 'tensorflow': 'tf', 'pytorch': 'pyt'}
           self.checkpoint_base = self._search_checkpoint_base(self.task, backends[self.backend])

        self.repo.put_result(self.task, self.exp_config, self.log,
                            checkpoint_base=self.checkpoint_base,
                            checkpoint_store=self.checkpoint_store,
                            print_fn=self.print_fn,
                            hostname=self.hostname,
                            username=self.username,
                            label=self.label)

    @staticmethod
    def _search_checkpoint_base(task, backend):
        """Finds if the checkpoint exists as a zip file or a bunch of files."""
        zip = "{}-model-{}-{}.zip".format(task, backend, os.getpid())
        non_zip = "{}-model-{}-{}".format(task, backend, os.getpid())
        print(zip)
        if os.path.exists(zip):
            return zip
        elif os.path.exists(".graph".format(non_zip)):
            return non_zip
        return None


def create_reporting_hook(**kwargs):
    return XPCtlReporting(**kwargs)

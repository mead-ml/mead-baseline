from __future__ import print_function

import os
import getpass
import socket
import json
from baseline.reporting import EpochReportingHook
from mead.utils import read_config_file_or_json
from baseline.reporting import register_reporting
from xpctl.client import Configuration
from xpctl.client.api import XpctlApi
from xpctl.client import ApiClient
from xpctl.client.rest import ApiException
from xpctl.helpers import to_swagger_experiment, store_model
from mead.utils import hash_config


@register_reporting(name='xpctl')
class XPCtlReporting(EpochReportingHook):
    def __init__(self, **kwargs):
        super(XPCtlReporting, self).__init__(**kwargs)
        # throw exception if the next three can't be read from kwargs
        self.api_url = kwargs['host']
        self.exp_config = read_config_file_or_json(kwargs['config_file'])
        self.task = kwargs['task']

        self.label = kwargs.get('label', None)
        self.username = kwargs.get('user', getpass.getuser())
        self.hostname = kwargs.get('host', socket.gethostname())
        self.checkpoint_base = None
        self.checkpoint_store = kwargs.get('checkpoint_store', '/data/model-checkpoints')
        self.save_model = kwargs.get('save_model', False) # optionally save the model
        config = Configuration(host=self.api_url)
        api_client = ApiClient(config)
        self.api = XpctlApi(api_client)

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
        try:
            result = self.api.put_result(
                self.task,
                to_swagger_experiment(self.task,
                                      self.exp_config,
                                      self.log,
                                      username=self.username,
                                      label=self.label,
                                      hostname=self.hostname,
                                      )
            )
            if result.response_type == 'failure':
                raise RuntimeError(result.message)
            else:
                print('result stored with experiment id', result.message)
            if self.save_model:
                eid = result.message
                backend = self.exp_config.get('backend', 'default')
                backends = {'default': 'tf', 'tensorflow': 'tf', 'pytorch': 'pyt'}
                self.checkpoint_base = self._search_checkpoint_base(self.task, backends[backend],
                                                                    self.exp_config.get('basedir'))
                if self.checkpoint_base is None:
                    raise RuntimeError('No checkpoint files found')
                result = store_model(checkpoint_base=self.checkpoint_base,
                                     config_sha1=hash_config(self.exp_config),
                                     checkpoint_store=self.checkpoint_store)
                if result is not None:
                    print('model stored at {}'.format(result))
                    update_result = self.api.update_property(self.task, eid, prop='checkpoint', value=result)
                    print(update_result.message)
                else:
                    raise RuntimeError('failed to store model at {}'.format(self.checkpoint_store))
        except ApiException as e:
            raise RuntimeError(json.loads(e.body)['detail'])

    @staticmethod
    def _search_checkpoint_base(task, backend, basedir=None):
        """Finds if the checkpoint exists as a zip file or a bunch of files."""
        if basedir is not None:
            zip = "{}-{}.zip".format(basedir, os.getpid())
            non_zip = "{}-{}".format(basedir, os.getpid())
        else:
            zip = "{}-model-{}-{}.zip".format(task, backend, os.getpid())
            non_zip = "{}-model-{}-{}".format(task, backend, os.getpid())
        if os.path.exists(zip):
            return zip
        elif os.path.exists(".graph".format(non_zip)):
            return non_zip
        return None


def create_reporting_hook(**kwargs):
    return XPCtlReporting(**kwargs)

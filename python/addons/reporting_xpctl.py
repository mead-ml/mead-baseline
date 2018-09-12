from baseline.reporting import ReportingHook
from xpctl.core import ExperimentRepo
from baseline.utils import read_config_file
import getpass
import socket
import os


class XPCtlReporting(ReportingHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cred = read_config_file(os.path.expanduser(kwargs['hook_setting']['cred']))
        self.exp_config = read_config_file(os.path.expanduser(kwargs['config_file']))
        self.task = self.exp_config['task']
        self.print_fn = print
        self.username = kwargs['hook_setting'].get('user', getpass.getuser())
        self.hostname = kwargs['hook_setting'].get('host', socket.gethostname())
        self.checkpoint_base = None
        self.checkpoint_store = kwargs['hook_setting'].get('checkpoint_store', '/data/model-checkpoints')
        self.save_model = kwargs['hook_setting'].get('save_model', False) # optionally save the model

        self.repo = ExperimentRepo().create_repo(**self.cred)
        self.log = []

    def step(self, metrics, tick, phase, tick_type=None, **kwargs):
        """Write intermediate results to a logging memory object that ll be pushed to the xpctl repo

        :param metrics: A map of metrics to scores
        :param tick: The time (resolution defined by `tick_type`)
        :param phase: The phase of training (`Train`, `Valid`, `Test`)
        :param tick_type: The resolution of tick (`STEP`, `EPOCH`)
        :return:
        """
        if tick_type is None:
            tick_type = 'STEP'
            if phase in ['Valid', 'Test']:
                tick_type = 'EPOCH'

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
                            username=self.username)

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

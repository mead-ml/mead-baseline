from __future__ import absolute_import, division, print_function, unicode_literals

import os
import platform
from baseline.utils import read_config_file
from baseline.utils import export as exporter
from mead.utils import get_mead_settings, modify_reporting_hook_settings
from hpctl.utils import hash_config


__all__ = []
export = exporter(__all__)


@export
class Experiment(object):
    """Object that holds experiment configuration.

    :param experiment: str, the name of the experiment.
    :param config: str, the path to the configuration file.
    :param logging: str, the path to the mead logging config file.
    :param hpctl_logging: str, the path to the hpctl logging file.
    :param settings: str, The path to the mead settings.
    :param datasets: str, The datasets file.
    :param embeddings: str, The embeddings file.
    :param task: str, The name of the task.
    """
    def __init__(
            self,
            experiment, config,
            logging, hpctl_logging,
            settings,
            datasets, embeddings, task,
            **kwargs
    ):
        super(Experiment, self).__init__()
        self.experiment_name = experiment
        self.hpctl_config = read_config_file(config)
        self.mead_config = self.hpctl_config['mead']
        if isinstance(self.mead_config, str):
            self.mead_config = read_config_file(self.mead_config)
        if kwargs.get('reporting') is not None:
            self.mead_config['reporting'] = kwargs.get('reporting')
        self.mead_logs = read_config_file(logging)
        self.hpctl_logs = read_config_file(hpctl_logging)
        self.mead_settings = get_mead_settings(settings)
        self.hpctl_settings = self.mead_settings.get('hpctl', {})
        self.hpctl_logs['host'] = self.hpctl_settings.get('logging', {}).get('host', 'localhost')
        self.hpctl_logs['port'] = self.hpctl_settings.get('logging', {}).get('port', 6006)
        self.datasets = datasets
        self.embeddings = embeddings
        self.task_name = task
        # The hash of the mead config (including sampling info) used to name
        # the results persistence.
        self.experiment_hash = hash_config(self.mead_config)
        try:
            os.mkdir(self.experiment_hash)
        except OSError:
            pass
        if self.experiment_name is not None:
            try:
                os.symlink(self.experiment_hash, self.experiment_name)
            except:
                pass
            print("Running experiment under name [{}]".format(self.experiment_name))
        ends = {}
        if kwargs.get('frontend') is None:
            ends['frontend'] = self.hpctl_config.get('frontend', self.hpctl_settings.get('frontend', {'type': 'console'}))
        else:
            ends['frontend'] = {'type': kwargs['frontend']}
        if kwargs.get('backend') is None:
            ends['backend'] = self.hpctl_config.get('backend', self.hpctl_settings.get('backend', {'type': 'mp'}))
        else:
            ends['backend'] = {'type': kwargs['backend']}
        modify_reporting_hook_settings(ends, kwargs['unknown'], {'backend', 'frontend'})
        self.frontend_config = ends['frontend']
        self.backend_config = ends['backend']
        if isinstance(self.backend_config.get('real_gpus'), str):
            self.backend_config['real_gpus'] = list(map(int, self.backend_config['real_gpus'].split(",")))

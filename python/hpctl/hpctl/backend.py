from __future__ import absolute_import, division, print_function, unicode_literals

import os
import requests
from baseline.utils import export as exporter
from hpctl.utils import Label
from hpctl.results import States


__all__ = []
export = exporter(__all__)


@export
def get_backend(backend_config):
    """Get the backend object.

    :param exp: hpctl.experiment.Experiment, The experiment data object.
    :param results: hpctl.results.Results, The data results object.
    """
    backend_type = backend_config['type']
    print("Using backend [{}]".format(backend_type))

    if backend_type == "mp":
        from hpctl.mp import MPBackend
        Be = MPBackend

    if backend_type == "docker":
        from hpctl.dock import DockerBackend
        Be = DockerBackend

    if backend_type == "remote":
        Be = RemoteBackend

    return Be(**backend_config)


@export
class Runner(object):
    """Abstract base class that handles running and stopping a single job."""
    def __init__(self, *args, **kwargs):
        super(Runner, self).__init__()
        self.p = None

    def start(self, exp, label, *args, **kwargs):
        pass

    def join(self):
        pass

    @property
    def is_done(self):
        pass

    def stop(self):
        pass


@export
class Backend(object):
    """Abstract class that handles running models."""
    def __init__(self):
        super(Backend, self).__init__()

    def launch(self, label, config, **kwargs):
        pass

    def any_done(self):
        pass

    def all_done(self):
        pass

    def kill(self, label):
        pass


class RemoteBackend(Backend):
    def __init__(self, host, port, **kwargs):
        super(RemoteBackend, self).__init__()
        self.host = host
        self.port = port
        self.labels = []

    def any_done(self):
        return True

    def all_done(self):
        # Track all label you personally launched and check if they are done.
        undone = []
        for label in self.labels:
            r = requests.post("http://{}:{}/hpctl/v1/state/{}/{}/{}".format(self.host, self.port, label.exp, label.sha1, label.human))
            if r.status_code != 200:
                return False
            if r.json()['state'] != str(States.DONE):
                undone.append(label)
        self.labels = undone
        return not self.labels

    def launch(self, **kwargs):
        kwargs['command'] = 'launch'
        self.labels.append(kwargs['label'])
        kwargs['label'] = str(kwargs['label'])
        r = requests.post("http://{}:{}/hpctl/v1/launch".format(self.host, self.port), json=kwargs)
        if r.status_code != 200:
            raise Exception


class LocalGPUBackend(Backend):
    def __init__(self, real_gpus=None, **kwargs):
        super(LocalGPUBackend, self).__init__()
        self.real_gpus = real_gpus
        if real_gpus is None:
            self.real_gpus = os.getenv("CUDA_VISIBLE_DEVICES", os.getenv("NV_GPU", "0")).split(',')
            print('read: {} from envs'.format(self.real_gpus))
        self.jobs = []
        self.label_to_job = {}
        self.gpus_to_job = {gpu: None for gpu in self.real_gpus}

    def launch(self, *args, **kwargs):
        pass

    def any_done(self):
        return (
            any(map(lambda x: x.is_done, self.jobs)) or
            any(map(lambda x: x is None, self.gpus_to_job.values()))
        )

    def all_done(self):
        return all(map(lambda x: x.is_done, self.jobs))

    def kill(self, label, results):
        pass

    def __del__(self):
        for job in self.jobs:
            job.join()

from __future__ import absolute_import, division, print_function, unicode_literals

import os
from baseline.utils import export as exporter
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
        from hpctl.remote import RemoteBackend
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
        self.labels = []

    def launch(self, label, config, **kwargs):
        pass

    def any_done(self):
        pass

    def all_done(self, results):
        # Track all label you personally launched and check if they are done.
        undone = []
        for label in self.labels:
            state = results.get_state(label)
            if state is not States.DONE and state is not States.KILLED:
                undone.append(label)
        self.labels = undone
        return not self.labels

    def kill(self, label):
        pass


class LocalGPUBackend(Backend):
    def __init__(self, real_gpus=None, **kwargs):
        super(LocalGPUBackend, self).__init__()
        self.real_gpus = real_gpus
        if real_gpus is None:
            self.real_gpus = os.getenv("CUDA_VISIBLE_DEVICES", os.getenv("NV_GPU", "0")).split(',')
            print('read: {} from envs'.format(self.real_gpus))
        self.real_gpus = map(str, self.real_gpus)
        self.jobs = []
        self.label_to_job = {}
        self.gpus_to_job = {gpu: None for gpu in self.real_gpus}

    def launch(self, *args, **kwargs):
        pass

    def _free_resources(self):
        for job in self.jobs:
            # Update label -> job mapping
            if job.is_done:
                to_del = None
                for l, cand_job in self.label_to_job.items():
                    if job == cand_job:
                        to_del = l
                if to_del is not None:
                    del self.label_to_job[to_del]

                # Free gpus
                for gpu, cand_job in self.gpus_to_job.items():
                    if job == cand_job:
                        self.gpus_to_job[gpu] = None

                job.join()
                self.jobs.remove(job)

    def _request_gpus(self, count):
        gpus = []
        for gpu, job in self.gpus_to_job.items():
            if job is None:
                gpus.append(str(gpu))
                if len(gpus) == count:
                    return gpus
        return

    def _reserve_gpus(self, gpus, job):
        for gpu in gpus:
            self.gpus_to_job[gpu] = job

    def any_done(self):
        return (
            any(map(lambda x: x.is_done, self.jobs)) or
            any(map(lambda x: x is None, self.gpus_to_job.values()))
        )


    def kill(self, label):
        if label not in self.label_to_job:
            return
        to_kill = self.label_to_job[label]
        to_kill.stop()
        to_kill.join()
        self._free_resources()

    def __del__(self):
        for job in self.jobs:
            job.join()

from __future__ import absolute_import, division, print_function, unicode_literals

import os
from subprocess import call
from baseline.utils import export as exporter
from baseline.utils import import_user_module
from hpctl.results import States


__all__ = []
export = exporter(__all__)


@export
def get_backend(backend_config):
    """Get the backend object.

    :param backend_config: dict, The arguments to initialize the backend.
    """
    backend_type = backend_config['type']
    print("Using backend [{}]".format(backend_type))

    if backend_type == "mp":
        from hpctl.mp import create_backend
    elif backend_type == "docker":
        from hpctl.dock import create_backend
    elif backend_type == "remote":
        from hpctl.remote import create_backend
    else:
        mod = import_user_module("backend", backend_type)
        create_backend = mod.create_backend

    return create_backend(**backend_config)


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

    @property
    def failed(self):
        pass

    def stop(self):
        pass


@export
class Backend(object):
    """Abstract class that handles running models."""
    def __init__(self):
        super(Backend, self).__init__()
        self.labels = []
        self.jobs = []
        self.label_to_job = {}

    def launch(self, label, *args, **kwargs):
        self.labels.append(label)

    def any_done(self):
        pass

    def _free_resources(self):
        pass

    def all_done(self, results):
        # Track all label you personally launched and check if they are done.
        undone = []
        for label in self.labels:
            job = self.label_to_job.get(label)
            if job is not None and job.is_done and job.failed:
                results.set_killed(label)
                self._free_resources()
            state = results.get_state(label)
            if state is not States.DONE and state is not States.KILLED:
                undone.append(label)
            else:
                self.label_to_job.pop(label, None)
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
        self.real_gpus = list(map(str, self.real_gpus))
        print("Running Jobs on the following GPU(s), {}".format(self.real_gpus))
        with open(os.devnull, 'w') as f:
            call('wall "HPCTL will run jobs on the following GPU(s) {}"'.format(self.real_gpus), shell=True, stdout=f, stderr=f)
        self.gpus_to_job = {gpu: None for gpu in self.real_gpus}

    def _free_resources(self):
        jobs = []
        for job in self.jobs:
            if job.is_done:

                # Free gpus
                for gpu, cand_job in self.gpus_to_job.items():
                    if job == cand_job:
                        self.gpus_to_job[gpu] = None

                job.join()
            else:
                jobs.append(job)
        self.jobs = jobs


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
        if not self.jobs:
            return True
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

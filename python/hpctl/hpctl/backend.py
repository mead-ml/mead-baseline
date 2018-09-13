from __future__ import absolute_import, division, print_function, unicode_literals

import os
from baseline.utils import export as exporter


__all__ = []
export = exporter(__all__)


@export
def handle_gpus(real_gpus, gpus, parallel_limit):
    """Function to handle gpu usage.

    :param gpus: List[str], The gpus.
    :param num_jobs: int, The number of concurrent jobs to run.
    """
    # If none given check in ENV varibales
    if real_gpus is None:
        real_gpus = os.getenv("CUDA_VISIBLE_DEVICES", os.getenv("NV_GPU", "0")).split(',')
        print('read: {} from envs'.format(real_gpus))
    real_gpus = list(map(str, real_gpus))
    if gpus is None:
        gpus = 1
    if gpus > len(real_gpus):
        if len(real_gpus) <= 1:
            raise RuntimeError("Asked to run each job on [{}] GPUs but only [{}] is available.".format(gpus, len(real_gpus)))
        raise RuntimeError("Asked to run each job on [{}] GPUs but only [{}] are available.".format(gpus, len(real_gpus)))
    # If the gpus can be split evenly across jobs then run jobs with multi_gpus
    real_gpus = [real_gpus[i:i+gpus] for i in range(0, min(len(real_gpus) // gpus + 1, len(real_gpus)), gpus)]
    if parallel_limit is not None:
        real_gpus = real_gpus[:parallel_limit]
    return real_gpus, gpus


@export
def get_backend(exp, results):
    """Get the backend object.

    :param exp: hpctl.experiment.Experiment, The experiment data object.
    :param results: hpctl.results.Results, The data results object.
    """
    backend = exp.backend_config
    backend_type = backend['type']
    print("Using backend [{}]".format(backend_type))

    gpus = exp.mead_config['model'].get('gpus', exp.hpctl_config.get('gpus', exp.hpctl_settings.get('gpus')))

    if backend_type == "mp":
        from hpctl.mp import MPBackend
        Be = MPBackend

    if backend_type == "docker":
        from hpctl.dock import DockerBackend
        Be = DockerBackend

    return Be(exp, results, gpus=gpus, **backend)


@export
class Runner(object):
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

    def launch(self, label, config):
        pass

    def any_done(self):
        pass

    def all_done(self):
        pass

    def kill(self, label):
        pass


@export
class LocalGPUBackend(Backend):
    def __init__(self, exp, results, gpus=None, real_gpus=None, parallel_limit=None, **kwargs):
        super(LocalGPUBackend, self).__init__()
        self.exp = exp
        self.results = results
        self.jobs = []
        self.label_to_job = {}
        self.real_gpus, self.gpus = handle_gpus(real_gpus, gpus, parallel_limit)
        if self.gpus == 1:
            print("Running each job on [{}] GPU".format(self.gpus))
        else:
            print("Running each job on [{}] GPUs".format(self.gpus))
        if len(self.real_gpus) == 1:
            print("Running [{}] job in parallel".format(len(self.real_gpus)))
        else:
            print("Running [{}] jobs in parallel".format(len(self.real_gpus)))

    def launch(self, label, config):
        """Launch a job.

        :param label: str, The name of the job.
        :param config: dict, The mead config for the job.
        """
        pass

    def any_done(self):
        return any(map(lambda x: x.is_done, self.jobs))

    def all_done(self):
        return all(map(lambda x: x.is_done, self.jobs))

    def kill(self, label):
        """Kill job with name `label`."""
        if not isinstance(label, Label):
            human, sha1 = self.results.get_label_prefix(label)
            if sha1:
                label = Label(sha1[0], human)
            else:
                return
        self.results.set_state(label, States.KILLED)
        job = self.label_to_job.get(label)
        if job is None:
            return
        job.stop()

    def __del__(self):
        for job in self.jobs:
            job.join()

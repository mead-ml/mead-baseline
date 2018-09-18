from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip, range

import os
import json
import docker
from baseline.utils import write_json, read_config_file
from hpctl.backend import LocalGPUBackend, handle_gpus, Runner
from hpctl.utils import create_logs


def create_mounts(default_mounts, user_mounts, cwd, datacache=None):
    """Create the mounting dict, Mounts are ro except for the cwd and datacache mount.

    :param default_mounts: List[str], The dirs to mount.
    :param user_mounts: List[str], The list of user defined places to mount.
    :param cwd: str, The dir to use as the working directory for docker.
    :param datacache: str, The location where datasets are downloaded to.

    :returns:
        dict, The mounting dict in the form {'external_loc', {'bind': 'internal_loc', 'mode': 'r[ow]'}}.
    """
    mounts = default_mounts + user_mounts
    modes = ['ro'] * len(mounts)
    modes.append('rw')  # for the cwd
    mounts.append(cwd)
    if datacache is not None:
        mounts.append(datacache)
        modes.append('rw')
    return create_mount_dict(mounts, modes)


def create_mount_dict(mount_dirs, modes):
    """Create the mounting dict.

    :param mount_dirs: List[str], The dirs to mount.
    :param modes: List[str], The mount modes.

    :returns:
        dict, The mounting dict in the form {'external_loc', {'bind': 'internal_loc', 'mode': 'r[ow]'}}
    ."""
    mounts = {}
    for mount, mode in zip(mount_dirs, modes):
        mounts[mount] = {"bind": mount, "mode": mode}
    return mounts


def get_container(framework):
    """Translate the framework into the docker container name."""
    return 'baseline-{}'.format({
        'tensorflow': 'tf',
        'pytorch': 'pyt',
        'dynet': 'dy'
    }[framework])


def run_docker(
        client, label, config_params,
        default_mounts=None, user_mounts=None,
        mead_logs=None, hpctl_logs=None,
        settings=None, task_name=None,
        datasets=None, embeddings=None,
        gpus=None, **kwargs
):
    """Run a model using docker.

    :param client: docker.Client, The docker client that talks to the docker daemon.
    :param label: hpctl.utils.Label, The label of the job.
    :param config_params: dict, The mead config.
    :param default_mounts: List[str], The dirs to mount.
    :param user_mounts: List[str], The user defined dirs to mount.
    :param mead_logs: dict, The mead logging config.
    :param hpctl_logs: dict, The hpctl logging config.
    :param settings: dict, The mead and hpctl settings.
    :param task_name: str, The name of the mead task.
    :param datasets: str, The dataset file.
    :param embeddings: str, The embeddings file.
    :param gpus: List[str], The gpus the job is allowed to use.

    :returns:
        tuple(docker.Container, str)
            The docker container to check on the status of the job,
            The working dir for the container.
    """
    loc = os.path.realpath(os.path.join(str(label.exp), label.local))
    curr = os.getcwd()
    try:
        os.makedirs(loc)
    except OSError:
        pass
    os.chdir(loc)

    cache = settings.get('datacache')

    # Write config files into working dir
    write_json(config_params, 'config.json')
    logs = create_logs(label, mead_logs, hpctl_logs)

    container = get_container(config_params['backend'])
    command = [
        'mead-train',
        '--config', '$CONFIG',
        '--settings', '$SETTINGS',
        '--datasets', '$DATASETS',
        '--embeddings', '$EMBEDDINGS',
        '--logging', '$LOGGING',
        '--task', task_name,
        '--gpus', str(len(gpus)),
    ]

    c = client.containers.run(
        container,
        command,
        runtime='nvidia',
        environment={
            'NV_GPU': ','.join(gpus),
            'CONFIG': json.dumps(config_params),
            'SETTINGS': json.dumps(settings),
            'DATASETS': json.dumps(read_config_file(datasets)),
            'EMBEDDINGS': json.dumps(read_config_file(embeddings)),
            'LOGGING': json.dumps(logs),
        },
        network_mode='host',
        working_dir=loc,
        volumes=create_mounts(default_mounts, user_mounts, loc, cache),
        detach=True,
    )
    os.chdir(curr)
    return c, loc


class DockerRunner(Runner):
    def __init__(self, client, func, gpus=None, *args, **kwargs):
        super(DockerRunner, self).__init__()
        self.func = func
        self.client = client
        self.gpus = gpus

    def start(self, exp, label, *args, **kwargs):
        kwargs['gpus'] = self.gpus
        # args = tuple([exp, label] + list(args))
        self.p, self.loc = self.func(self.client, exp, label, *args, **kwargs)
        while self.is_done:
            pass

    def join(self):
        if self.p is None:
            return
        self.p.wait()
        # Dump everything the docker container outputs to a file.
        with open(os.path.join(self.loc, 'stdout'), 'wb') as f:
            f.write(self.p.logs())

    @property
    def is_done(self):
        if self.p is None:
            return True
        self.p.reload()
        return not self.p.status == 'running'

    def stop(self):
        if self.p is None:
            return
        self.p.kill()


class DockerBackend(LocalGPUBackend):
    """Backend that launches docker jobs.

    :param results: hpctl.results.Results, The results storage location.
    :param default_mounts: List[str], The dirs to mount.
    :param user_mounts: List[str], The user dirs to mount.
    """
    def __init__(self, exp, results, default_mounts=None, user_mounts=None, **kwargs):
        super(DockerBackend, self).__init__(exp, results, **kwargs)
        self.client = docker.from_env()
        self.jobs = [DockerRunner(self.client, run_docker, gpu) for gpu in self.real_gpus]
        self.default_mounts = kwargs.get('default_mounts', [])
        self.user_mounts = kwargs.get('user_mounts', [])

    def launch(self, label, config):
        """Start a job.

        :param label: hpctl.utils.Label, The label for the job.
        :param config: dict, the config for the model.
        """
        for job in self.jobs:
            # update label -> job mapping.
            if job.is_done:
                to_del = None
                for l, cand_job in self.label_to_job.items():
                    if job == cand_job:
                        to_del = l
                if to_del is not None:
                    del self.label_to_job[l]
                job.join()
                job.start(
                    label, config,
                    self.default_mounts, self.user_mounts,
                    mead_logs=self.exp.mead_logs,
                    hpctl_logs=self.exp.hpctl_logs,
                    settings=self.exp.mead_settings,
                    datasets=self.exp.datasets,
                    embeddings=self.exp.embeddings,
                    task_name=self.exp.task_name
                )
                self.label_to_job[label] = job
                return

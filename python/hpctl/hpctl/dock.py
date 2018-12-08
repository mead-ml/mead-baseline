from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip, range

import os
import json
import docker
from baseline.utils import export as exporter
from baseline.utils import write_json, read_config_file
from hpctl.utils import create_logs
from hpctl.backend import LocalGPUBackend, Runner, register_backend


__all__ = []
export = exporter(__all__)


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


def get_container_name(framework):
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
    :param datasets: dict, The dataset mappings.
    :param embeddings: dict, The embeddings mappings.
    :param gpus: List[str], The gpus the job is allowed to use.

    :returns:
        tuple(docker.Container, str)
            The docker container to check on the status of the job,
            The working dir for the container.
    """
    loc = os.path.realpath(os.path.join(label.exp, label.sha1, label.name))
    curr = os.getcwd()
    try:
        os.makedirs(loc)
    except OSError:
        pass
    os.chdir(loc)

    cache = os.path.expanduser(settings.get('datacache'))

    # Write config files into working dir
    write_json(config_params, 'config.json')
    logs = create_logs(label, mead_logs, hpctl_logs)

    if 'visdom' in config_params.get('reporting', {}):
        config_params.get('reporting', {})['visdom']['name'] = label.name

    container = get_container_name(config_params['backend'])
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
            'DATASETS': json.dumps(datasets),
            'EMBEDDINGS': json.dumps(embeddings),
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
    def __init__(self, client):
        super(DockerRunner, self).__init__()
        self.p = None
        self.client = client

    def start(self, func, *args, **kwargs):
        self.p, self.loc = func(self.client, *args, **kwargs)
        while self.is_done:
            pass

    def join(self):
        if self.p is None:
            return
        self.p.wait()
        # Dump everything the docker container outputs to a file.
        with open(os.path.join(self.loc, 'stdout'), 'wb') as f:
            f.write(self.p.logs())
        self.p.remove()
        self.p = None

    @property
    def is_done(self):
        if self.p is None:
            return True
        self.p.reload()
        return not self.p.status == 'running'

    @property
    def failed(self):
        return False if self.p is None else self.client.api.inspect_container(self.p.id)['State']['ExitCode'] != 0

    def stop(self):
        if self.p is None:
            return
        self.p.kill()


@export
@register_backend('docker')
class DockerBackend(LocalGPUBackend):
    """Backend that launches docker jobs.

    :param default_mounts: List[str], The dirs to mount.
    :param user_mounts: List[str], The user dirs to mount.
    """
    def __init__(self, default_mounts=None, user_mounts=None, **kwargs):
        super(DockerBackend, self).__init__(**kwargs)
        self.client = docker.from_env()
        self.default_mounts = default_mounts if default_mounts is not None else []
        self.user_mounts = user_mounts if user_mounts is not None else []

    def launch(
            self,
            label, config,
            mead_logs, hpctl_logs,
            settings, datasets,
            embeddings, task_name,
            **kwargs
    ):
        """Start a job.

        :param label: hpctl.utils.Label, The label for the job.
        :param config: dict, the config for the model.
        :param exp: hpctl.experiment.Experiment, The experiment data object.
        """
        super(DockerBackend, self).launch(label)
        self._free_resources()
        gpu = self._request_gpus(1)

        job = DockerRunner(self.client)
        job.start(
            run_docker,
            label, config,
            self.default_mounts, self.user_mounts,
            mead_logs=mead_logs,
            hpctl_logs=hpctl_logs,
            settings=settings,
            datasets=datasets,
            embeddings=embeddings,
            task_name=task_name,
            gpus=gpu
        )
        self.label_to_job[label] = job
        self._reserve_gpus(gpu, job)
        self.jobs.append(job)


@export
def create_backend(**kwargs):
    return DockerBackend(**kwargs)

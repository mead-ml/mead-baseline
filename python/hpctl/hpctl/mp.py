from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip, map, range

import os
import sys
import json
import traceback
from multiprocessing import Process
from subprocess import check_call, call, CalledProcessError
import mead
from baseline.utils import write_json
from baseline.utils import export as exporter
from hpctl.results import States
from hpctl.utils import create_logs, Label
from hpctl.backend import LocalGPUBackend, handle_gpus, Runner


try:
    from setproctitle import setproctitle
except ImportError:
    setproctitle = lambda x: None

__all__ = []
export = exporter(__all__)


def run_job(
        label,
        config_params,
        mead_logs=None, hpctl_logs=None,
        settings=None, task_name=None,
        datasets=None, embeddings=None,
        gpus=None, **kwargs
):
    """Function that runs a meed job.

    :param label: Label, The Label (sha1 and human name) of the model.
    :param config_params: dict, The config for the job.
    :param mead_logs: dict, The mead logging config.
    :param hpctl_logs: dict, The hpctl logging config.
    :param settings: str, The location of the mead settings file.
    :param task_name: str, The name of the mead task.
    :param datasets: str, The location of the dataset file.
    :param embeddings: str, The location of the embeddings file.
    :param gpus: List[str], The list of gpus the process is allowed to use.
    """
    # Suppress tensorflow CUDA output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
    config_params['visdom_name'] = label.human
    config_params['model']['gpus'] = len(gpus)

    write_json(config_params, 'config.json')
    logs = create_logs(label, mead_logs, hpctl_logs)
    task = mead.Task.get_task_specific(task_name, logs, settings)
    # Change this when PR#131 is merged: remove task_name
    task.read_config(config_params, datasets, task_name)
    task.initialize(embeddings)
    task.train()
    os.remove(logs)


@export
class FileProcess(Process):
    """A process that writes all stdout to a file.

    Output is written to `exp/label/stdout`

    :param exp: str, The name of the experiment.
    :param label: Label, The Label (sha1 and human name) of the model.
    """
    def __init__(self, exp, label, *args, **kwargs):
        super(FileProcess, self).__init__(*args, **kwargs)
        self.exp = exp
        self.label = label.sha1
        self.human = label.human
        self.loc = os.path.join(exp, str(label))
        try:
            os.makedirs(self.loc)
        except OSError:
            pass
        self.out_file = os.path.join(self.loc, 'stdout')
        self.output = open(self.out_file, 'w', buffering=1)

    def run(self):
        out, err = sys.stdout, sys.stderr
        sys.stdout = self.output
        sys.stderr = self.output
        os.chdir(self.loc)
        setproctitle(self.human)
        try:
            super(FileProcess, self).run()
        except Exception as e:
            sys.stderr = err
            sys.stdout = out
            raise(e)

    def join(self):
        super(FileProcess, self).join()
        self.output.close()


@export
class TmuxProcess(FileProcess):
    """A process that writes all stdout to a file and sets up tmux to look at it.

    Output is written to `exp/label/stdout`
    Use tmux with `tmux attach -t human_label`

    :param exp: str, The name of the experiment.
    :param label: Label, The Label (sha1 and human name) of the model.
    """
    def __init__(self, *args, **kwargs):
        super(TmuxProcess, self).__init__(*args, **kwargs)
        cmd = 'tail -f {}'.format(self.out_file)
        with open(os.devnull, 'w') as devnull:
            try:
                _ = check_call('tmux -V', shell=True, stdout=devnull, stderr=devnull)
                self.tmux = True
            except CalledProcessError:
                self.tmux = False
            if self.tmux:
                # tmux new-window -n {name} -d would add a window but it only
                # add the window to the most recent session. So if you use tmux
                # to look at a different experiment then new jobs from this one
                # would be added to that session. Once I figure out how to ping
                # a session from python we can have this use exp as the session
                # name and human in the window, if the call errors we can ping
                # the session and then call new-window. Probably need a lock
                # so these can't step on each other.
                call('tmux new-sess -s {} -n {} -d {}'.format(
                    self.human, self.human, cmd
                ), shell=True, stdout=devnull, stderr=devnull)

    def join(self):
        super(TmuxProcess, self).join()
        if self.tmux:
            with open(os.devnull, 'w') as devnull:
                call('tmux kill-session -t {}'.format(self.human),
                     shell=True, stdout=devnull, stderr=devnull
                )


class MPRunner(Runner):
    """Wrapper around run job to make interfacing easier.

    Each job holds it own gpus so I don't have to track which are used when
    launching a new job.

    :param func: Callable, The function you run.
    :param gpus: List[str], The gpus you can use.
    """
    def __init__(self, func, gpus=None, *args, **kwargs):
        super(Runner, self).__init__()
        self.p = None
        self.func = func
        self.gpus = gpus

    def start(self, exp, label, *args, **kwargs):
        kwargs['gpus'] = self.gpus
        args = tuple([label] + list(args))
        self.p = TmuxProcess(exp, label, target=self.func, args=args, kwargs=kwargs)
        self.p.start()
        while self.is_done:
            pass

    def join(self):
        if self.p is None:
            return
        self.p.join()

    @property
    def is_done(self):
        if self.p is None:
            return True
        return not self.p.is_alive()

    def stop(self):
        self.p.terminate()


@export
class MPBackend(LocalGPUBackend):
    """Back end that runs multiprocessing jobs.

    :param exp: hpctl.experiment.Experiment: The experiment config.
    :param results: hpctl.results.Results: The data storage object.
    :param num_jobs: int, The number of concurrent jobs to run.
    :param gpus: List[str], The gpus.
    """
    def __init__(
            self,
            exp, results,
            **kwargs
    ):
        super(MPBackend, self).__init__(exp, results, **kwargs)
        self.jobs = [MPRunner(run_job, gpu) for gpu in self.real_gpus]

    def launch(self, label, config):
        """Start a job.

        :param label: hpctl.utils.Label, The label for the job.
        :param config: dict, the config for the model.
        """
        exp = self.exp.experiment_hash
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
                    exp, label, config,
                    mead_logs=self.exp.mead_logs,
                    hpctl_logs=self.exp.hpctl_logs,
                    settings=self.exp.mead_settings,
                    datasets=self.exp.datasets,
                    embeddings=self.exp.embeddings,
                    task_name=self.exp.task_name
                )
                self.label_to_job[label] = job
                return

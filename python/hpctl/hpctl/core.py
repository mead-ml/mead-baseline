from __future__ import absolute_import, division, print_function, unicode_literals

import os
from baseline.utils import export as exporter
from baseline.utils import read_config_file, write_json, hash_config
from hpctl.results import Results
from hpctl.backend import get_backend
from hpctl.logging_server import Logs
from hpctl.experiment import Experiment
from hpctl.sample import get_config_sampler
from hpctl.frontend import get_frontend, color
from hpctl.utils import create_logs
from hpctl.scheduler import RoundRobinScheduler


__all__ = []
export = exporter(__all__)


# def serve(**kwargs):
#     """Spin up a flask server. This might become a thing that is used in the server version of hpctl."""
#     exp = Experiment(**kwargs)
#     config = read_config_file(kwargs['config'])
#     config_hash = hash_config(config['mead'])
#     results = Results.create(config_hash)
#     from flask import Flask
#     from hpctl.flask_frontend import init_app, FlaskFrontend
#     from multiprocessing import Queue
#     app = Flask(__name__)
#     fe = FlaskFrontend(Queue, exp, results)
#     init_app(app, fe)
#     app.run(debug=kwargs['debug'])



@export
def list_names(**kwargs):
    """List all the human names from an experiment. For easy navigating afterwards."""
    config = read_config_file(kwargs['config'])
    config_hash = hash_config(config['mead'])
    results = Results.create(config_hash)
    for label in results.get_labels():
        print("{} {}".format(
            color(results.get_state(label)),
            label.human
        ))


@export
def find(**kwargs):
    """Find the location of job information based on human name."""
    name = kwargs['name']
    config = read_config_file(kwargs['config'])
    config_hash = hash_config(config['mead'])
    results = Results.create(config_hash)
    s = results._getvalue()
    # Look in human to label
    human, sha1 = results.get_label_prefix(name)
    if human is not None:
        print("{} ->".format(human))
        for sha in sha1:
            print("\t{}".format(os.path.join(config_hash, sha)))
        return
    # Look in label to human
    sha1, human = results.get_human_prefix(name)
    if sha1 is not None:
        print("{} ->".format(sha1))
        for h in human:
              print("\t{}".format(h))
        return
    print("Can't find {} in {}".format(name, config_hash))


from hpctl.frontend import Console
class DummyFrontend(Console):
    def __init__(self, exp, results, count):
        super(DummyFrontend, self).__init__(exp, results)
        self.count = count
        self.config_sampler = get_config_sampler(
            exp.mead_config, None, exp.hpctl_config.get('samplers', [])
        )
        self.cs = [self.config_sampler.sample() for _ in range(self.count)]
        self.cs[0][0].exp = "EXAMPLE"
        for x in self.cs:
            print(x[0])
        print()

    def command(self):
        if self.count >= 0:
            self.count -= 1
            if self.count == 0:
                from copy import deepcopy
                exp = deepcopy(self.exp)
                exp.experiment_hash = "EXAMPLE"
            else:
                exp = self.exp
            return {
                "command": "start",
                "label": self.cs[self.count][0],
                "config": self.cs[self.count][1],
                "exp": exp
            }
        return None, None

def serve(**kwargs):
    # temp
    exp = Experiment(**kwargs)
    results = Results.create()
    backend = get_backend(exp)
    logs = Logs.create(exp)
    frontend = DummyFrontend(exp, results, 4)
    scheduler = RoundRobinScheduler()
    try:
        run_forever(results, backend, scheduler, frontend, logs)
    except KeyboardInterrupt:
        pass



@export
def search(**kwargs):
    """Search for optimal hyperparameters."""
    exp = Experiment(**kwargs)

    results = Results.create()

    backend = get_backend(exp)

    # Setup the sampler
    config_sampler = get_config_sampler(
        exp.mead_config,
        results,
        exp.hpctl_config.get('samplers', [])
    )

    logs = Logs.create(exp)

    frontend = get_frontend(exp, results)

    num_iters = int(kwargs.get('num_iters') if kwargs.get('num_iters') is not None else exp.hpctl_config.get('num_iters', 3))

    run(num_iters, exp, results, backend, frontend, config_sampler, logs)
    logs.stop()
    frontend.finalize()
    results.save()


@export
def run(num_iters, exp, results, backend, frontend, config_sampler, logs):
    """The main driver of hpctl.

    :param num_iters: int, The number of jobs to run.
    :param exp: hpctl.experiment.Experiment: The experiment config.
    :param results: hpctl.results.Results: The data storage object.
    :param backend: hpctl.backend.Backend, The job launcher.
    :param frontend: hpctl.frontend.Frontent, The result displayer.
    :param config_sampler: hpctl.sample.ConfigSampler, The object to generate
        model configs.
    :param logs: hpctl.logging_server.Logs, The log collector.
    """
    launched = 0
    all_done = False
    while not all_done:
        # Launch jobs
        if backend.any_done() and launched < num_iters:
            label, config = config_sampler.sample()
            results.insert(label, config)
            results.save()
            backend.launch(label, config, exp)
            frontend.update()
            launched += 1
        # Monitor jobs
        label, message = logs.get()
        if label is not None:
            results.update(label, message)
            results.save()
            frontend.update()
        # Get user inputs
        cmd = frontend.command()
        process_command(cmd, backend, frontend, None, results)
        # Check for quit
        all_done = backend.all_done() if launched >= num_iters else False


def run_forever(results, backend, scheduler, frontend, logs):
    while True:
        cmd = frontend.command()
        process_command(cmd, backend, frontend, scheduler, logs)
        if backend.any_done():
            exp_hash, job_blob = scheduler.get()
            if exp_hash is not None:
                results.insert(job_blob['label'], job_blob['config'])
                results.save()
                backend.launch(**job_blob)
                frontend.update()
        # Monitor jobs
        label, message = logs.get()
        if label is not None:
            results.update(label, message)
            results.save()
            frontend.update()


def process_command(cmd, backend, frontend, scheduler, results):
    if cmd is not None and isinstance(cmd, dict):
        if cmd['command'] == 'kill':
            backend.kill(cmd['label'], results)
            frontend.update()
        if cmd['command'] == 'start':
            scheduler.add(cmd['label'].exp, cmd)

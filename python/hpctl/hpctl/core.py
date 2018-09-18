from __future__ import absolute_import, division, print_function, unicode_literals

import os
from baseline.utils import export as exporter
from baseline.utils import read_config_file, write_json
from hpctl.results import Results
from hpctl.backend import get_backend
from hpctl.logging_server import Logs
from hpctl.experiment import Experiment
from hpctl.sample import get_config_sampler
from hpctl.frontend import get_frontend, color
from hpctl.utils import create_logs, hash_config


__all__ = []
export = exporter(__all__)


def serve(**kwargs):
    """Spin up a flask server. This might become a thing that is used in the server version of hpctl."""
    exp = Experiment(**kwargs)
    config = read_config_file(kwargs['config'])
    config_hash = hash_config(config['mead'])
    results = Results.create(config_hash)
    from flask import Flask
    from hpctl.flask_frontend import init_app, FlaskFrontend
    from multiprocessing import Queue
    app = Flask(__name__)
    fe = FlaskFrontend(Queue, exp, results)
    init_app(app, fe)
    app.run(debug=kwargs['debug'])


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


@export
def search(**kwargs):
    """Search for optimal hyperparameters."""
    exp = Experiment(**kwargs)
    results = Results.create(exp.experiment_hash)

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
    results.save(exp.experiment_hash)


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
            results.save(exp.experiment_hash)
            backend.launch(label, config, exp)
            frontend.update()
            launched += 1
        # Monitor jobs
        label, message = logs.get()
        if label is not None:
            results.update(label, message)
            results.save(exp.experiment_hash)
            frontend.update()
        # Get user inputs
        cmd = frontend.command()
        process_command(cmd, backend, frontend, results)
        # Check for quit
        all_done = backend.all_done() if launched >= num_iters else False


def process_command(cmd, backend, frontend, results):
    if cmd is not None and isinstance(cmd, dict):
        if cmd['command'] == 'kill':
            backend.kill(cmd['label'], results)
            frontend.update()

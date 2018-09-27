from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
from baseline.utils import export as exporter
from baseline.utils import read_config_file, write_json, hash_config
from mead.utils import read_config_file_or_json
from hpctl.utils import create_logs
from hpctl.results import get_results
from hpctl.backend import get_backend
from hpctl.sample import get_config_sampler
from hpctl.frontend import get_frontend, color
from hpctl.logging_server import get_log_server
from hpctl.scheduler import RoundRobinScheduler
from hpctl.settings import (
    get_configs,
    get_settings,
    get_logs,
    get_ends,
    set_root,
)


__all__ = []
export = exporter(__all__)


@export
def list_names(**kwargs):
    """List all the human names from an experiment. For easy navigating afterwards."""
    config = read_config_file(kwargs['config'])
    config_hash = hash_config(config['mead'])
    results = Results.create(config_hash)
    for label in results.get_labels():
        print("{} {}".format(
            color(results.get_state(label)),
            label.name
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
def launch(**kwargs):
    hp_config, mead_config = get_configs(**kwargs)
    exp_hash = hash_config(mead_config)
    hp_settings, mead_settings = get_settings(**kwargs)
    hp_logs, mead_logs = get_logs(hp_settings, **kwargs)
    datasets = read_config_file_or_json(kwargs['datasets'])
    embeddings = read_config_file_or_json(kwargs['embeddings'])
    task_name = kwargs.get('task', mead_config.get('task', 'None'))
    _, backend_config = get_ends(hp_config, hp_settings, **kwargs)
    backend_config['type'] = 'remote'
    if 'host' not in backend_config:
        backend_config['host'] = 'localhost'
    if 'port' not in backend_config:
        backend_config['port'] = 5000

    config_sampler = get_config_sampler(mead_config, None, hp_config.get('samplers', []))
    label, config = config_sampler.sample()
    print(label)
    send = {
        'label': str(label),
        'config': config,
        'datasets': datasets,
        'embeddings': embeddings,
        'mead_logs': mead_logs,
        'hpctl_logs': hp_logs,
        'task_name': task_name,
        'settings': mead_settings,
        'experiment_config': mead_config,
    }

    be = get_backend(backend_config)
    be.launch(**send)


@export
def serve(**kwargs):
    hp_settings, mead_settings = get_settings(**kwargs)
    frontend_config, backend_config = get_ends({}, hp_settings, **kwargs)
    hp_logs, _ = get_logs(hp_settings, **kwargs)
    set_root(hp_settings)

    results = get_results({})
    backend = get_backend(backend_config)
    logs = get_log_server(hp_logs)
    frontend_config['type'] = 'flask'
    frontend = get_frontend(frontend_config, results)
    scheduler = RoundRobinScheduler()
    try:
        run_forever(results, backend, scheduler, frontend, logs)
    except KeyboardInterrupt:
        pass


def _remote_monkey_patch(backend_config, hp_logs, results_config):
    if backend_config['type'] == 'remote':
        hp_logs['type'] = 'remote'
        results_config['type'] = 'remote'
        results_config['host'] = backend_config['host']
        results_config['port'] = backend_config['port']
    return backend_config, hp_logs, results_config


@export
def search(**kwargs):
    """Search for optimal hyperparameters."""
    hp_config, mead_config = get_configs(**kwargs)
    exp_hash = hash_config(mead_config)
    hp_settings, mead_settings = get_settings(**kwargs)
    hp_logs, mead_logs = get_logs(hp_settings, **kwargs)
    datasets = read_config_file_or_json(kwargs['datasets'])
    embeddings = read_config_file_or_json(kwargs['embeddings'])
    task_name = kwargs.get('task', mead_config.get('task', 'None'))
    frontend_config, backend_config = get_ends(hp_config, hp_settings, **kwargs)

    frontend_config['experiment_hash'] = exp_hash
    default = mead_config['train'].get('early_stopping_metric', 'avg_loss')
    if 'train' not in frontend_config:
        frontend_config['train'] = default
    if 'dev' not in frontend_config:
        frontend_config['dev'] = default
    if 'test' not in frontend_config:
        frontend_config['test'] = default

    set_root(hp_settings)
    backend_config, hp_logs, results_config = _remote_monkey_patch(backend_config, hp_logs, {})

    results = get_results(results_config)
    results.add_experiment(mead_config)

    backend = get_backend(backend_config)

    # Setup the sampler
    config_sampler = get_config_sampler(
        mead_config,
        results,
        hp_config.get('samplers', [])
    )

    logs = get_log_server(hp_logs)

    frontend = get_frontend(frontend_config, results)

    num_iters = int(kwargs.get('num_iters') if kwargs.get('num_iters') is not None else hp_config.get('num_iters', 3))

    run(num_iters, results, backend, frontend, config_sampler, logs, mead_logs, hp_logs, mead_settings, datasets, embeddings, task_name)
    logs.stop()
    frontend.finalize()
    results.save()


@export
def run(num_iters, results, backend, frontend, config_sampler, logs, mead_logs, hpctl_logs, mead_settings, datasets, embeddings, task_name):
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
            backend.launch(
                label=label, config=config,
                mead_logs=mead_logs, hpctl_logs=hpctl_logs,
                settings=mead_settings, datasets=datasets,
                embeddings=embeddings, task_name=task_name
            )
            results.set_running(label)
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
        time.sleep(1)


def run_forever(results, backend, scheduler, frontend, logs):
    while True:
        cmd = frontend.command()
        process_command(cmd, backend, frontend, scheduler, results)
        if backend.any_done():
            exp_hash, job_blob = scheduler.get()
            if exp_hash is not None:
                backend.launch(**job_blob)
                results.set_running(job_blob['label'])
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
            print(cmd)
            backend.kill(cmd['label'], results)
            results.set_killed(cmd['label'])
            frontend.update()
        if cmd['command'] == 'launch':
            exp_config = cmd.pop('experiment_config', None)
            if exp_config is not None:
                results.add_experiment(exp_config)
            scheduler.add(cmd['label'].exp, cmd)
            results.insert(cmd['label'], cmd['config'])
            results.save()

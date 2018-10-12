from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import json
from baseline.utils import export as exporter
from baseline.utils import read_config_file, write_json
from mead.utils import read_config_file_or_json, hash_config, parse_extra_args, index_by_label
from hpctl.report import get_xpctl
from hpctl.utils import create_logs
from hpctl.results import get_results
from hpctl.backend import get_backend
from hpctl.sample import get_config_sampler
from hpctl.logging_server import get_log_server
from hpctl.scheduler import RoundRobinScheduler
from hpctl.frontend import get_frontend, color_state
from hpctl.settings import (
    get_config,
    get_settings,
    get_logs,
    get_ends,
    set_root,
    get_xpctl_settings,
)


__all__ = []
export = exporter(__all__)


@export
def list_names(root, **kwargs):
    """List all the human names from an experiment. For easy navigating afterwards."""
    results = get_results({"file_name": os.path.join(root, 'results')})
    experiments = results.get_experiments()
    for experiment in experiments:
        print(experiment)
        for label in results.get_labels(experiment):
            state = results.get_state(label)
            print("\t{state} {sha1} -> {name}".format(state=color_state(state), **label))


@export
def find(name, root, **kwargs):
    """Find the location of job information based on human name."""
    results = get_results({"file_name": os.path.join(root, 'results')})
    _, label = results.get_label_prefix(name)
    if label is not None:
        print(label)
    else:
        print("Can't find {}".format(name))


def force_remote_backend(backend_config):
    backend_config['type'] = 'remote'
    backend_config.setdefault('host', 'localhost')
    backend_config.setdefault('port', 5000)


@export
def launch(
        config, settings,
        logging, hpctl_logging,
        datasets, embeddings,
        reporting, unknown,
        task, num_iters, **kwargs
):
    mead_config = get_config(config, reporting, unknown)
    exp_hash = hash_config(mead_config)
    hp_settings, mead_settings = get_settings(settings)
    hp_logs, mead_logs = get_logs(hp_settings, logging, hpctl_logging)
    datasets = read_config_file_or_json(datasets)
    embeddings = read_config_file_or_json(embeddings)
    if task is None:
        task = mead_config.get('task', 'classify')
    _, backend_config = get_ends(hp_settings, unknown)

    force_remote_backend(backend_config)

    config_sampler = get_config_sampler(mead_config, None)
    be = get_backend(backend_config)

    for _ in range(num_iters):
        label, config = config_sampler.sample()
        print(label)
        send = {
            'label': label,
            'config': config,
            'mead_logs': mead_logs,
            'hpctl_logs': hp_logs,
            'task_name': task,
            'settings': mead_settings,
            'experiment_config': mead_config,
        }
        be.launch(**send)


@export
def serve(settings, hpctl_logging, embeddings, datasets, unknown, **kwargs):
    hp_settings, mead_settings = get_settings(settings)
    frontend_config, backend_config = get_ends(hp_settings, unknown)
    hp_logs, _ = get_logs(hp_settings, {}, hpctl_logging)
    xpctl_config = get_xpctl_settings(mead_settings)
    set_root(hp_settings)

    datasets = read_config_file_or_json(datasets)
    embeddings = read_config_file_or_json(embeddings)

    results = get_results({})
    backend = get_backend(backend_config)
    logs = get_log_server(hp_logs)

    xpctl = get_xpctl(xpctl_config)

    frontend_config['type'] = 'flask'
    frontend_config['datasets'] = index_by_label(datasets)
    frontend_config['embeddings'] = index_by_label(embeddings)
    frontend = get_frontend(frontend_config, results, xpctl)
    scheduler = RoundRobinScheduler()

    cache = mead_settings.get('datacache', '~/.bl-data')

    try:
        run_forever(results, backend, scheduler, frontend, logs, cache, xpctl_config, datasets, embeddings)
    except KeyboardInterrupt:
        pass


def _remote_monkey_patch(backend_config, hp_logs, results_config, xpctl_config):
    if backend_config.get('type', 'local') == 'remote':
        hp_logs['type'] = 'remote'
        results_config['type'] = 'remote'
        results_config['host'] = backend_config['host']
        results_config['port'] = backend_config['port']
        if xpctl_config is not None:
            xpctl_config['type'] = 'remote'
            xpctl_config['host'] = backend_config['host']
            xpctl_config['port'] = backend_config['port']


@export
def search(
        config, settings,
        logging, hpctl_logging,
        datasets, embeddings,
        reporting, unknown,
        task, num_iters,
        **kwargs
):
    """Search for optimal hyperparameters."""
    mead_config = get_config(config, reporting, unknown)
    exp_hash = hash_config(mead_config)

    hp_settings, mead_settings = get_settings(settings)

    hp_logs, mead_logs = get_logs(hp_settings, logging, hpctl_logging)

    datasets = read_config_file_or_json(datasets)
    embeddings = read_config_file_or_json(embeddings)

    if task is None:
        task = mead_config.get('task', 'classify')

    frontend_config, backend_config = get_ends(hp_settings, unknown)

    # Figure out xpctl
    xpctl_config = None
    auto_xpctl = 'xpctl' in mead_config.get('reporting', [])
    if not auto_xpctl:
        # If the jobs aren't setup to use xpctl automatically create your own
        xpctl_config = get_xpctl_settings(mead_settings)
        if xpctl_config is not None:
            xpctl_extra = parse_extra_args(['xpctl'], unknown)
            xpctl_config['label'] = xpctl_extra.get('xpctl', {}).get('label')
    results_config = {}

    # Set frontend defaults
    frontend_config['experiment_hash'] = exp_hash
    default = mead_config['train'].get('early_stopping_metric', 'avg_loss')
    frontend_config.setdefault('train', 'avg_loss')
    frontend_config.setdefault('dev', default)
    frontend_config.setdefault('test', default)

    # Negotiate remote status
    if backend_config['type'] != 'remote':
        set_root(hp_settings)
    _remote_monkey_patch(backend_config, hp_logs, results_config, xpctl_config)

    xpctl = get_xpctl(xpctl_config)

    results = get_results(results_config)
    results.add_experiment(mead_config)

    backend = get_backend(backend_config)

    config_sampler = get_config_sampler(mead_config, results)

    logs = get_log_server(hp_logs)

    frontend = get_frontend(frontend_config, results, xpctl)

    labels = run(num_iters, results, backend, frontend, config_sampler, logs, mead_logs, hp_logs, mead_settings, datasets, embeddings, task)
    logs.stop()
    frontend.finalize()
    results.save()
    if auto_xpctl:
        for label in labels:
            results.set_xpctl(label, True)
    return labels, results


def force_xpctl(mead_config, label):
    xpctl = mead_config.setdefault('reporting', {}).setdefault('xpctl', {})
    if label is not None:
        xpctl['label'] = label


@export
def verify(
        config, settings,
        logging, hpctl_logging,
        datasets, embeddings,
        reporting, unknown,
        task, num_iters, label,
        **kwargs
):
    """Run an experiment while forcing xpctl on used to run a single config a lot."""
    mead_config = get_config(config, reporting, unknown)

    force_xpctl(mead_config, label)

    return search(config, settings, logging, hpctl_loggins, datasets, embeddings, reporting, unknown, task, num_iters)


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
    launched_labels = []
    all_done = False
    while not all_done:
        # Launch jobs
        if backend.any_done() and launched < num_iters:
            label, config = config_sampler.sample()
            results.insert(label, config)
            results.save()
            launched_labels.append(label)
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
        process_command(cmd, backend, frontend, None, results, frontend.xpctl)
        # Check for quit
        all_done = backend.all_done(results) if launched >= num_iters else False
        time.sleep(1)
    return launched_labels


def override_client_settings(settings, cache, xpctl_config):
    """Use server POV for things line cache and xpctl config."""
    settings['datacache'] = cache
    if xpctl_config is None:
        settings.get('reporting', {}).pop('xpctl', None)
    else:
        settings.get('reporting', {}).get('xpctl', {})['cred'] = xpctl_config


def run_forever(results, backend, scheduler, frontend, logs, cache, xpctl_config, datasets, embeddings):
    while True:
        cmd = frontend.command()
        process_command(cmd, backend, frontend, scheduler, results, None)
        if backend.any_done():
            exp_hash, job_blob = scheduler.get()
            if exp_hash is not None:
                label, job = job_blob
                override_client_settings(job['settings'], cache, xpctl_config)
                job['embeddings'] = embeddings
                job['datasets'] = datasets
                backend.launch(**job)
                results.set_running(label)
                frontend.update()
        # Monitor jobs
        label, message = logs.get()
        if label is not None:
            results.update(label, message)
            results.save()
            frontend.update()
        backend.all_done(results)


def process_command(cmd, backend, frontend, scheduler, results, xpctl):
    if cmd is not None and isinstance(cmd, dict):
        # Kill a job
        if cmd['command'] == 'kill':
            backend.kill(cmd['label'])
            results.set_killed(cmd['label'])
            if scheduler is not None:
                scheduler.remove(cmd['label'])
            frontend.update()
        # Start a job
        if cmd['command'] == 'launch':
            exp_config = cmd.pop('experiment_config', None)
            if exp_config is not None:
                results.add_experiment(exp_config)
            scheduler.add(cmd['label'], cmd)
            results.insert(cmd['label'], cmd['config'])
            results.save()
        # Save a run to XPCTL
        if cmd['command'] == 'xpctl':
            if xpctl is not None and not results.get_xpctl(cmd['label']):
                id_ = xpctl.put_result(cmd['label'])
                results.set_xpctl(cmd['label'], id_)

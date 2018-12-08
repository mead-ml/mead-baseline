import argparse
from copy import deepcopy
from itertools import chain
from baseline.utils import read_config_stream
import mead
from mead.utils import convert_path, parse_extra_args, normalize_backend


def main():
    parser = argparse.ArgumentParser(description='Train a text classifier')
    parser.add_argument('--config', help='JSON Configuration for an experiment', type=convert_path, default="$MEAD_CONFIG")
    parser.add_argument('--settings', help='JSON Configuration for mead', default='config/mead-settings.json', type=convert_path)
    parser.add_argument('--datasets', help='json library of dataset labels', default='config/datasets.json', type=convert_path)
    parser.add_argument('--embeddings', help='json library of embeddings', default='config/embeddings.json', type=convert_path)
    parser.add_argument('--logging', help='json file for logging', default='config/logging.json', type=convert_path)
    parser.add_argument('--task', help='task to run', choices=['classify', 'tagger', 'seq2seq', 'lm'])
    parser.add_argument('--gpus', help='Number of GPUs (defaults to number available)', type=int, default=-1)
    parser.add_argument('--reporting', help='reporting hooks', nargs='+')
    parser.add_argument('--backend', help='The deep learning backend to use')
    args, reporting_args = parser.parse_known_args()

    config_params = read_config_stream(args.config)
    try:
        args.settings = read_config_stream(args.settings)
    except:
        print('Warning: no mead-settings file was found at [{}]'.format(args.config))
        args.settings = {}
    args.datasets = read_config_stream(args.datasets)
    args.embeddings = read_config_stream(args.embeddings)
    args.logging = read_config_stream(args.logging)

    if args.gpus is not None:
        config_params['model']['gpus'] = args.gpus

    if args.backend is not None:
        config_params['backend'] = normalize_backend(args.backend)

    cmd_hooks = args.reporting if args.reporting is not None else []
    config_hooks = config_params.get('reporting') if config_params.get('reporting') is not None else []
    reporting = parse_extra_args(set(chain(cmd_hooks, config_hooks)), reporting_args)
    config_params['reporting'] = reporting

    task_name = config_params.get('task', 'classify') if args.task is None else args.task
    print('Task: [{}]'.format(task_name))
    task = mead.Task.get_task_specific(task_name, args.logging, args.settings)
    task.read_config(config_params, args.datasets, reporting_args=reporting_args, config_file=deepcopy(config_params))
    task.initialize(args.embeddings)
    task.train()


if __name__ == "__main__":
    main()

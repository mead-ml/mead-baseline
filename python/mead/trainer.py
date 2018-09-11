import argparse
import mead
from mead.utils import convert_path
from baseline.utils import read_config_file


def main():
    parser = argparse.ArgumentParser(description='Train a text classifier')
    parser.add_argument('--config', help='JSON Configuration for an experiment', required=True, type=convert_path)
    parser.add_argument('--settings', help='JSON Configuration for mead', default='config/mead-settings.json', type=convert_path)
    parser.add_argument('--datasets', help='json library of dataset labels', default='config/datasets.json', type=convert_path)
    parser.add_argument('--embeddings', help='json library of embeddings', default='config/embeddings.json', type=convert_path)
    parser.add_argument('--logging', help='json file for logging', default='config/logging.json', type=convert_path)
    parser.add_argument('--task', help='task to run', choices=['classify', 'tagger', 'seq2seq', 'lm'])
    parser.add_argument('--gpus', help='Number of GPUs (defaults to 1)', type=int)
    parser.add_argument('--reporting', help='reporting hooks', nargs='+')
    args = parser.parse_known_args()[0]

    config_params = read_config_file(args.config)

    if args.gpus is not None:
        config_params['model']['gpus'] = args.gpus
    if args.reporting is not None:
        config_params['reporting'] = args.reporting

    task_name = config_params.get('task', 'classify') if args.task is None else args.task
    print('Task: [{}]'.format(task_name))
    task = mead.Task.get_task_specific(task_name, args.logging, args.settings)
    task.read_config(config_params, args.config, args.datasets, task_name)
    task.initialize(args.embeddings)
    task.train()


if __name__ == "__main__":
    main()

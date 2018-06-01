import argparse
import mead
from mead.utils import convert_path


def main():
    parser = argparse.ArgumentParser(description='Train a text classifier')
    parser.add_argument('--config', help='JSON Configuration for an experiment', required=True, type=convert_path)
    parser.add_argument('--settings', help='JSON Configuration for mead', default='config/mead-settings.json', type=convert_path)
    parser.add_argument('--datasets', help='json library of dataset labels', default='config/datasets.json', type=convert_path)
    parser.add_argument('--embeddings', help='json library of embeddings', default='config/embeddings.json', type=convert_path)
    parser.add_argument('--logging', help='json file for logging', default='config/logging.json', type=convert_path)
    parser.add_argument('--task', help='task to run', default='classify', choices=['classify', 'tagger', 'seq2seq', 'lm'])
    args = parser.parse_known_args()[0]

    task = mead.Task.get_task_specific(args.task, args.logging, args.settings)
    task.read_config(args.config, args.datasets)
    task.initialize(args.embeddings)
    task.train()


if __name__ == "__main__":
    main()

import argparse
import mead
from mead.utils import convert_path


def main():
    parser = argparse.ArgumentParser(description='Train a text classifier')
    parser.add_argument('--config', help='JSON Configuration for an experiment', required=True, type=convert_path)
    parser.add_argument('--settings', help='JSON Configuration for mead', required=False, default='config/mead-settings.json', type=convert_path)
    parser.add_argument('--datasets', help='json library of dataset labels', default='config/datasets.json', type=convert_path)
    parser.add_argument('--embeddings', help='json library of embeddings', default='config/embeddings.json', type=convert_path)
    parser.add_argument('--logging', help='json file for logging', default='config/logging.json', type=convert_path)
    parser.add_argument('--task', help='task to run', default='classify', choices=['classify', 'tagger', 'seq2seq', 'lm'])
    parser.add_argument('--model', help='model name', required=True)
    parser.add_argument('--model_version', help='model_version', default=1)
    parser.add_argument('--output_dir', help='output dir', default='./models')
    args = parser.parse_args()

    task = mead.Task.get_task_specific(args.task, args.logging, args.settings)
    task.read_config(args.config, args.datasets)

    exporter = task.create_exporter()
    exporter.run(args.model, args.embeddings, args.output_dir, args.model_version)


if __name__ == "__main__":
    main()

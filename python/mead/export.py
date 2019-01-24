import argparse
import mead
from mead.utils import convert_path
from baseline.utils import unzip_model
from baseline.utils import read_config_file
from mead.exporters import create_exporter


def main():
    parser = argparse.ArgumentParser(description='Export a model')
    parser.add_argument('--config', help='JSON Configuration for an experiment', required=True, type=convert_path)
    parser.add_argument('--settings', help='JSON Configuration for mead', required=False, default='config/mead-settings.json', type=convert_path)
    parser.add_argument('--datasets', help='json library of dataset labels', default='config/datasets.json', type=convert_path)
    parser.add_argument('--logging', help='json file for logging', default='config/logging.json', type=convert_path)
    parser.add_argument('--task', help='task to run', choices=['classify', 'tagger', 'seq2seq', 'lm'])
    parser.add_argument('--exporter_type', help='exporter type', default='default')
    parser.add_argument('--model', help='model name', required=True, type=unzip_model)
    parser.add_argument('--model_version', help='model_version', default=1)
    parser.add_argument('--output_dir', help='output dir', default='./models')
    parser.add_argument('--beam', help='beam_width', default=30, type=int)
    parser.add_argument('--is_remote', help='if True, separate items for remote server and client. If False bundle everything together', default=True, type=lambda x: (str(x).lower() == 'true'))

    args = parser.parse_args()

    config_params = read_config_file(args.config)

    task_name = config_params.get('task', 'classify') if args.task is None else args.task

    if task_name == 'seq2seq' and 'beam' not in config_params:
        config_params['beam'] = args.beam

    task = mead.Task.get_task_specific(task_name, args.logging, args.settings)
    task.read_config(config_params, args.datasets)
    exporter = create_exporter(task, args.exporter_type)
    exporter.run(args.model, args.output_dir, args.model_version, remote=args.is_remote)

if __name__ == "__main__":
    main()

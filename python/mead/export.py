import os
import logging
import argparse
from baseline.utils import unzip_files, read_config_stream
import mead
from mead.exporters import create_exporter
from mead.utils import (
    convert_path,
    configure_logger,
    get_export_params,
    create_feature_exporter_field_map,
)


DEFAULT_SETTINGS_LOC = 'config/mead-settings.json'
DEFAULT_LOGGING_LOC = 'config/logging.json'
logger = logging.getLogger('mead')


def main():
    parser = argparse.ArgumentParser(description='Export a model')
    parser.add_argument('--config', help='configuration for an experiment', required=True, type=convert_path)
    parser.add_argument('--settings', help='configuration for mead', required=False, default=DEFAULT_SETTINGS_LOC, type=convert_path)
    parser.add_argument('--modules', help='modules to load', default=[], nargs='+', required=False)
    parser.add_argument('--logging', help='json file for logging', default=DEFAULT_LOGGING_LOC, type=convert_path)
    parser.add_argument('--task', help='task to run', choices=['classify', 'tagger', 'seq2seq', 'lm'])
    parser.add_argument('--exporter_type', help="exporter type (default 'default')", default=None)
    parser.add_argument('--return_labels', help='if true, the exported model returns actual labels else '
                                                'the indices for labels vocab (default False)', default=None)
    parser.add_argument('--model', help='model name', required=True, type=unzip_files)
    parser.add_argument('--model_version', help='model_version', default=None)
    parser.add_argument('--output_dir', help="output dir (default './models')", default=None)
    parser.add_argument('--project', help='Name of project, used in path first', default=None)
    parser.add_argument('--name', help='Name of the model, used second in the path', default=None)
    parser.add_argument('--beam', help='beam_width', default=30, type=int)
    parser.add_argument('--is_remote', help='if True, separate items for remote server and client. If False bundle everything together (default True)', default=None)

    args = parser.parse_args()
    configure_logger(args.logging)

    config_params = read_config_stream(args.config)

    try:
        args.settings = read_config_stream(args.settings)
    except Exception:
        logger.warning('Warning: no mead-settings file was found at [{}]'.format(args.settings))
        args.settings = {}

    task_name = config_params.get('task', 'classify') if args.task is None else args.task

    # Remove multigpu references
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    os.environ['NV_GPU'] = ""
    if 'gpus' in config_params.get('train', {}):
        del config_params['train']['gpus']

    if task_name == 'seq2seq' and 'beam' not in config_params:
         config_params['beam'] = args.beam

    config_params['modules'] = config_params.get('modules', []) + args.modules

    task = mead.Task.get_task_specific(task_name, args.settings)

    output_dir, project, name, model_version, exporter_type, return_labels, is_remote = get_export_params(
        config_params.get('export', {}),
        args.output_dir,
        args.project, args.name,
        args.model_version,
        args.exporter_type,
        args.return_labels,
        args.is_remote,
    )
    # Here we reuse code in `.read_config` which needs a dataset index (when used with mead-train)
    # but when used with mead-export it is not needed. This is a dummy dataset index that will work
    # It means we don't need to pass it in
    datasets = [{'label': config_params['dataset']}]
    task.read_config(config_params, datasets, exporter_type=exporter_type)
    feature_exporter_field_map = create_feature_exporter_field_map(config_params['features'])
    exporter = create_exporter(task, exporter_type, return_labels=return_labels,
                               feature_exporter_field_map=feature_exporter_field_map)
    exporter.run(args.model, output_dir, project, name, model_version, remote=is_remote)


if __name__ == "__main__":
    main()

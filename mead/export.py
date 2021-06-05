import os
import logging
import argparse
from itertools import chain
from baseline.utils import (
    unzip_files,
    str2bool,
    read_config_file,
    normalize_backend,
    read_config_stream
)

import mead
from mead.exporters import create_exporter
from mead.utils import (
    convert_path,
    configure_logger,
    get_export_params,
    parse_extra_args,
    create_feature_exporter_field_map,
    parse_and_merge_overrides,
)


DEFAULT_SETTINGS_LOC = 'config/mead-settings.json'
DEFAULT_LOGGING_LOC = 'config/logging.json'
logger = logging.getLogger('mead')


def main():
    parser = argparse.ArgumentParser(description='Export a model')
    parser.add_argument('--config', help='configuration for an experiment', required=True, type=convert_path)
    parser.add_argument('--settings', help='configuration for mead', required=False, default=DEFAULT_SETTINGS_LOC, type=convert_path)
    parser.add_argument('--modules', help='modules to load', default=[], nargs='+', required=False)
    parser.add_argument('--datasets', help='json library of dataset labels')
    parser.add_argument('--vecs', help='index of vectorizers: local file, remote URL or hub mead-ml/ref', default='config/vecs.json', type=convert_path)
    parser.add_argument('--logging', help='json file for logging', default='config/logging.json', type=convert_path)
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
    parser.add_argument('--nbest_input', help='Is the input to this model N-best', default=False, type=str2bool)
    parser.add_argument('--is_remote', help='if True, separate items for remote server and client. If False bundle everything together (default True)', default=None)
    parser.add_argument('--backend', help='The deep learning backend to use')
    parser.add_argument('--reporting', help='reporting hooks', nargs='+')
    parser.add_argument('--use_version', help='Should we use the version?', type=str2bool, default=True)
    parser.add_argument('--zip', help='Should we zip the results?', type=str2bool, default=False)

    args, overrides = parser.parse_known_args()
    configure_logger(args.logging)

    config_params = read_config_stream(args.config)
    config_params = parse_and_merge_overrides(config_params, overrides, pre='x')

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
    if args.backend is not None:
        config_params['backend'] = normalize_backend(args.backend)

    cmd_hooks = args.reporting if args.reporting is not None else []
    config_hooks = config_params.get('reporting') if config_params.get('reporting') is not None else []
    reporting = parse_extra_args(set(chain(cmd_hooks, config_hooks)), overrides)
    config_params['reporting'] = reporting

    args.vecs = read_config_stream(args.vecs)

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
    task.read_config(config_params, datasets, args.vecs, exporter_type=exporter_type)
    feature_exporter_field_map = create_feature_exporter_field_map(config_params['features'])
    exporter = create_exporter(task, exporter_type, return_labels=return_labels,
                               feature_exporter_field_map=feature_exporter_field_map,
                               nbest_input=args.nbest_input)
    exporter.run(args.model, output_dir, project, name, model_version, remote=is_remote, use_version=args.use_version, zip_results=args.zip)


if __name__ == "__main__":
    main()

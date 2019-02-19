import os
import logging
import argparse
from baseline.utils import unzip_files, read_config_file
import mead
from mead.exporters import create_exporter
from mead.utils import convert_path, configure_logger, get_export_params


logger = logging.getLogger('mead')


def create_feature_exporter_field_map(feature_section, default_exporter_field='tokens'):
    feature_exporter_field_map = {}
    for feature_desc in feature_section:
        if feature_desc.get('exporter_field') is None:
            feature_exporter_field_map[feature_desc['name']] = default_exporter_field
        else:
            feature_exporter_field_map[feature_desc['name']] = feature_desc['exporter_field']
    return feature_exporter_field_map


def main():
    parser = argparse.ArgumentParser(description='Export a model')
    parser.add_argument('--config', help='JSON Configuration for an experiment', required=True, type=convert_path)
    parser.add_argument('--settings', help='JSON Configuration for mead', required=False, default='config/mead-settings.json', type=convert_path)
    parser.add_argument('--modules', help='modules to load', default=[], nargs='+', required=False)
    parser.add_argument('--datasets', help='json library of dataset labels', default='config/datasets.json', type=convert_path)
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
    parser.add_argument('--is_remote', help='if True, separate items for remote server and client. If False bundle everything together (default True)', default=None)

    args = parser.parse_args()
    configure_logger(args.logging)

    config_params = read_config_file(args.config)

    try:
        args.settings = read_config_stream(args.settings)
    except:
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
    task.read_config(config_params, args.datasets, exporter_type=exporter_type)
    feature_exporter_field_map = create_feature_exporter_field_map(config_params['features'])
    exporter = create_exporter(task, exporter_type, return_labels=return_labels,
                               feature_exporter_field_map=feature_exporter_field_map)
    exporter.run(args.model, output_dir, project, name, model_version, remote=is_remote)


if __name__ == "__main__":
    main()

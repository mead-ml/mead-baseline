import os
import logging
import argparse
import copy
from itertools import chain
from baseline.utils import read_config_stream, normalize_backend
import mead
from mead.utils import convert_path, parse_extra_args, configure_logger

DEFAULT_SETTINGS_LOC = 'config/mead-settings.json'
DEFAULT_DATASETS_LOC = 'config/datasets.json'
DEFAULT_LOGGING_LOC = 'config/logging.json'
DEFAULT_EMBEDDINGS_LOC = 'config/embeddings.json'
logger = logging.getLogger('mead')


def update_datasets(datasets_config, config_params, train, valid, test):
    """Take an existing datasets index file and update to include a record with train/valid/test overrides

    If the label provided in the dataset is found in the dataset index, it will use that as a template and
    individually override the provided `train`, `valid` and `test` params to update that record.

    If the label does not exist, it creates a dummy record and augments that record with the provided `train`,
    `valid`, and optionally, the `test`

    :param datasets_config: The datasets config to update
    :param config_params: The mead config
    :param train: (`str`) An override train set or `None`. If `dataset` key doesnt exist, cannot be `None`
    :param valid: (`str`) An override valid set or `None` If `dataset` key doesnt exist, cannot be `None`
    :param test: (`str`) An override test set or None
    :return: None
    """

    for file_name in [train, valid, test]:
        if not os.path.exists(train):
            raise Exception('No such file exists for override: {}'.format(file_name))

    original_dataset_label = config_params['dataset']

    original_record = [entry for entry in datasets_config if entry['label'] == original_dataset_label]
    if not original_record:
        if not train or not valid:
            raise Exception('No template label provided, so you must provide at least train and valid params!')
        updated_record = {'label': original_record, 'train_file': None, 'valid_file': None, 'test_file': None}
    else:
        if len(original_record) != 1:
            logger.warning('Warning: multiple templates found for dataset override, using first!')
        updated_record = copy.deepcopy(original_record[0])
        if 'sha1' in updated_record:
            logger.info('Ignoring SHA1 due to user override')
            del updated_record['sha1']
        if 'download' in updated_record:
            if not train or not valid:
                raise Exception('Cannot override downloadable dataset without providing file '
                                'locations for both training and validation')
            if not test and 'test_file' in updated_record:
                del updated_record['test_file']
            del updated_record['download']
    new_dataset_label = '{}.{}'.format(original_dataset_label, os.getpid())
    updated_record['label'] = new_dataset_label

    if train:
        updated_record['train_file'] = train
    if valid:
        updated_record['valid_file'] = valid
    if test:
        updated_record['test_file'] = test

    logger.warning(updated_record)
    config_params['dataset'] = new_dataset_label
    logger.info("The dataset key for this override is {}".format(new_dataset_label))
    datasets_config.append(updated_record)


def main():
    parser = argparse.ArgumentParser(description='Train a text classifier')
    parser.add_argument('--config', help='configuration for an experiment', type=convert_path, default="$MEAD_CONFIG")
    parser.add_argument('--settings', help='configuration for mead', default=DEFAULT_SETTINGS_LOC, type=convert_path)
    parser.add_argument('--datasets', help='index of dataset labels', type=convert_path)
    parser.add_argument('--modules', help='modules to load', default=[], nargs='+', required=False)
    parser.add_argument('--mod_train_file', help='override the training set')
    parser.add_argument('--mod_valid_file', help='override the validation set')
    parser.add_argument('--mod_test_file', help='override the test set')
    parser.add_argument('--embeddings', help='index of embeddings', type=convert_path)
    parser.add_argument('--logging', help='config file for logging', default=DEFAULT_LOGGING_LOC, type=convert_path)
    parser.add_argument('--task', help='task to run', choices=['classify', 'tagger', 'seq2seq', 'lm'])
    parser.add_argument('--gpus', help='Number of GPUs (defaults to number available)', type=int, default=-1)
    parser.add_argument('--basedir', help='Override the base directory where models are stored', type=str)
    parser.add_argument('--reporting', help='reporting hooks', nargs='+')
    parser.add_argument('--backend', help='The deep learning backend to use')
    parser.add_argument('--checkpoint', help='Restart training from this checkpoint')
    args, reporting_args = parser.parse_known_args()

    config_params = read_config_stream(args.config)

    if args.basedir is not None:
        config_params['basedir'] = args.basedir

    task_name = config_params.get('task', 'classify') if args.task is None else args.task

    args.logging = read_config_stream(args.logging)
    configure_logger(args.logging, config_params.get('basedir', './{}'.format(task_name)))

    try:
        args.settings = read_config_stream(args.settings)
    except:
        logger.warning('Warning: no mead-settings file was found at [{}]'.format(args.settings))
        args.settings = {}

    args.datasets = args.datasets if args.datasets else args.settings.get('datasets', convert_path(DEFAULT_DATASETS_LOC))
    args.datasets = read_config_stream(args.datasets)
    if args.mod_train_file or args.mod_valid_file or args.mod_test_file:
        logging.warning('Warning: overriding the training/valid/test data with user-specified files'
                        ' different from what was specified in the dataset index.  Creating a new key for this entry')
        update_datasets(args.datasets, config_params, args.mod_train_file, args.mod_valid_file, args.mod_test_file)

    args.embeddings = args.embeddings if args.embeddings else args.settings.get('embeddings', convert_path(DEFAULT_EMBEDDINGS_LOC))
    args.embeddings = read_config_stream(args.embeddings)

    if args.gpus is not None:
        config_params['model']['gpus'] = args.gpus

    if args.backend is None and 'backend' in args.settings:
        args.backend = args.settings['backend']
    if args.backend is not None:
        config_params['backend'] = normalize_backend(args.backend)

    config_params['modules'] = list(set(chain(config_params.get('modules', []), args.modules)))

    cmd_hooks = args.reporting if args.reporting is not None else []
    config_hooks = config_params.get('reporting') if config_params.get('reporting') is not None else []
    reporting = parse_extra_args(set(chain(cmd_hooks, config_hooks)), reporting_args)
    config_params['reporting'] = reporting

    logger.info('Task: [{}]'.format(task_name))
    task = mead.Task.get_task_specific(task_name, args.settings)
    task.read_config(config_params, args.datasets, reporting_args=reporting_args)
    task.initialize(args.embeddings)
    task.train(args.checkpoint)


if __name__ == "__main__":
    main()

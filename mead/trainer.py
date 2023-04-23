import os
import logging
import argparse
import copy
from itertools import chain
from eight_mile.utils import read_config_stream, str2bool
from baseline.utils import import_user_module, normalize_backend
import mead
import hydra
from omegaconf import DictConfig
from mead.utils import convert_path, parse_extra_args, configure_logger, parse_and_merge_overrides

DEFAULT_SETTINGS_LOC = 'config/mead-settings.json'
DEFAULT_DATASETS_LOC = 'config/datasets.json'
DEFAULT_LOGGING_LOC = 'config/logging.json'
DEFAULT_EMBEDDINGS_LOC = 'config/embeddings.json'
DEFAULT_VECTORIZERS_LOC = 'config/vecs.json'

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


@hydra.main(version_base=None, config_path='./config', config_name="sst2-pyt")
def main(cfg: DictConfig):

    # task_module overrides are not allowed via hub or HTTP, must be defined locally
    if 'task_modules' in cfg:
        for task in cfg.task_modules:
            import_user_module(task)

    task_name = cfg.get('task', 'classify')
    #args.logging = read_config_stream(args.logging)
    #configure_logger(args.logging, config_params.get('basedir', './{}'.format(task_name)))
    logger.info('Task: [{}]'.format(task_name))

    task = mead.Task.get_task_specific(task_name, cfg.get('settings', {}))
    task.init(cfg)
    task.train()

if __name__ == "__main__":
    main()

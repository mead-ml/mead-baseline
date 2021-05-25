import os
import json
import shutil
import logging
import logging.config
import hashlib
from typing import Optional, List, Dict, Callable
from datetime import datetime
import argparse
from copy import deepcopy
from itertools import chain
from collections import OrderedDict, MutableMapping
from baseline.utils import exporter, str2bool, read_config_file, write_json, get_logging_level, validate_url, zip_files, delete_old_copy

__all__ = []
export = exporter(__all__)
logger = logging.getLogger('mead')

KNOWN_DATE_FMTS = [
    '%Y%m%d',
    '%Y-%m-%d',
    '%Y/%m/%d',
    '%Y',
    '%Y%m',
    '%Y-%m',
    '%Y/%m',
    '%Y%m%d_%H%M',
    '%Y-%m-%d_%H-%M',
    '%Y/%m/%d_%M/%H'
]


@export
def configure_logger(logger_config, basedir=None):
    """Use the logger file (logging.json) to configure the log, but overwrite the filename to include the PID

    There are reporting and timing loggers that are configured, the latter being used for speed testing.

    :param logger_config: The logging configuration JSON or file containing JSON
    :return: A dictionary config derived from the logger_file, with the reporting handler suffixed with PID
    """

    config = read_config_file_or_json(logger_config, 'logger')
    config['handlers']['reporting_file_handler']['filename'] = 'reporting-{}.log'.format(os.getpid())
    config['handlers']['timing_file_handler']['filename'] = 'timing-{}.log'.format(os.getpid())
    if basedir is not None:
        # Create a copy of the file handler so we can keep the log in cwd for back compat
        config['handlers']['reporting_basedir_handler'] = deepcopy(config['handlers']['reporting_file_handler'])
        # Update the file to be in the basedir of the model
        config['handlers']['reporting_basedir_handler']['filename'] = os.path.join(basedir, config['handlers']['reporting_basedir_handler']['filename'])
        config['handlers']['reporting_basedir_handler']['class'] = 'baseline.utils.MakeFileHandler'
        # Add this handler to the logger
        config['loggers']['baseline.reporting']['handlers'].append('reporting_basedir_handler')
        # Same as above
        config['handlers']['timing_basedir_handler'] = deepcopy(config['handlers']['timing_file_handler'])
        config['handlers']['timing_basedir_handler']['filename'] = os.path.join(basedir, config['handlers']['timing_basedir_handler']['filename'])
        config['handlers']['timing_basedir_handler']['class'] = 'baseline.utils.MakeFileHandler'
        config['loggers']['baseline.timing']['handlers'].append('timing_basedir_handler')
    level = os.getenv('LOG_LEVEL', 'INFO')
    config['loggers']['baseline']['level'] = get_logging_level(os.getenv('BASELINE_LOG_LEVEL', level))
    config['loggers']['mead']['level'] = get_logging_level(os.getenv('MEAD_LOG_LEVEL', level))
    config['handlers']['reporting_console_handler']['level'] = get_logging_level(os.getenv('REPORTING_LOG_LEVEL', level))
    config['handlers']['timing_console_handler']['level'] = get_logging_level(os.getenv('TIMING_LOG_LEVEL', level))

    logging.config.dictConfig(config)


@export
def print_dataset_info(dataset):
    logger.info("[train file]: {}".format(dataset['train_file']))
    logger.info("[valid file]: {}".format(dataset['valid_file']))
    if 'test_file' in dataset:
        logger.info("[test file]: {}".format(dataset['test_file']))
    vocab_file = dataset.get('vocab_file')
    if vocab_file is not None:
        logger.info("[vocab file]: {}".format(vocab_file))
    label_file = dataset.get('label_file')
    if label_file is not None:
        logger.info("[label file]: {}".format(label_file))


@export
def read_config_file_or_json(config, name=''):
    if isinstance(config, (dict, list)):
        return config
    config = os.path.expanduser(config)
    if os.path.exists(config):
        return read_config_file(config)
    raise Exception('Expected {} config file or a JSON object.'.format(name))


def parse_date(s, known_fmts: List[str] = KNOWN_DATE_FMTS):
    for fmt in known_fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt
        except:
            continue
    raise Exception("Couldn't parse datestamp {}".format(s))


@export
def flatten(dictionary, sep='.'):
    """Flatten a nested dict.

    :param dictionary: the dictionary to flatten.
    :param sep: The separator between old nested keys.
    :returns: The flattened dict.
    """

    def _flatten(dictionary, sep, prev):
        """This is the recursive function that actually does the work of flattening..

        :param dictionary: the dictionary to flatten.
        :param sep: The separator between old nested keys.
        :param prev: A recurrent state param that tracks key above you.
        :returns: The flattened dict.
        """
        flat = {}
        prev = [] if prev is None else prev
        for k, v in dictionary.items():
            if isinstance(v, MutableMapping):
                flat.update(_flatten(v, sep=sep, prev=list(chain(prev, [k]))))
            else:
                flat[sep.join(chain(prev, [k]))] = v
        return flat

    return _flatten(dictionary, sep, [])


@export
def unflatten(dictionary, sep: str = "."):
    """Turn a flattened dict into a nested dict.

    :param dictionary: The dict to unflatten.
    :param sep: This character represents a nesting level in a flattened key
    :returns: The nested dict
    """
    nested = {}
    for k, v in dictionary.items():
        keys = k.split(sep)
        it = nested
        for key in keys[:-1]:
            # If key is in `it` we get the value otherwise we get a new dict.
            # .setdefault will also set the new dict to the value of `it[key]`.
            # assigning to `it` will move us a step deeper into the nested dict.
            it = it.setdefault(key, {})
        it[keys[-1]] = v
    return nested


def _infer_numeric_or_str(value: str):
    """Convert value to an int, float or leave it as a string.

    :param value: The cli value to parse
    :returns: The value converted to int or float if possible
    """
    for func in (int, float):
        try:
            return func(value)
        except ValueError:
            continue
    return value


@export
def parse_overrides(overrides, pre):
    """Find override parameters in the cli args.

    Note:
        If you use the same cli flag multiple time the values will be
        aggregated into a list. For example
        `--x:a 1 --x:a 2 --x:a 3` will give back `{'a': [1, 2, 3]}`

    :param overrides: The cli flags and values.
    :param pre: only look at keys that start with `--{pre}:`
    :returns: The key value pairs from the command line args.
    """
    pre = f'--{pre}:'
    parser = argparse.ArgumentParser()
    for key in set(filter(lambda x: pre in x, overrides)):
        # Append action collect each value into a list allowing us to override a
        # yaml list by repeating the key.
        parser.add_argument(key, action='append', type=_infer_numeric_or_str)
    args = parser.parse_known_args(overrides)[0]
    return {k.split(":")[1]: v[0] if len(v) == 1 else v for k, v in vars(args).items()}


@export
def parse_and_merge_overrides(base, overrides, pre):
    """Parse extra cli args and use them to override the original config.

    :param base: The base config that will be overridden
    :param overrides: The cli args holding override values
    :param pre: The key used to find cli flags.
    :returns: The base config overridden with values from overrides.
    """
    overrides = parse_overrides(overrides, pre)
    base = flatten(base)
    base.update(overrides)
    return unflatten(base)


@export
def get_dataset_from_key(dataset_key, datasets_set):

    # This is the previous behavior
    if dataset_key in datasets_set:
        return datasets_set[dataset_key]

    last_date = parse_date('1900')
    last_k = None
    for k, v in datasets_set.items():

        if dataset_key in k:
            try:
                parts = k.split(':')
                dt = parse_date(parts[-1])
                dataset_name = ":".join(parts[:-1])
            except Exception:
                continue
            if dt > last_date and dataset_name == dataset_key:
                last_date = dt
                last_k = k

    if last_k is None:
        raise Exception("No dataset could be found with key {}".format(dataset_key))

    return datasets_set[last_k]


@export
def get_mead_settings(mead_settings_config):
    if mead_settings_config is None:
        return {}
    return read_config_file_or_json(mead_settings_config, 'mead settings')


@export
def index_by_label(object_list):
    objects = {x['label']: x for x in object_list}
    return objects


@export
def convert_path(path, loc=None):
    """If the provided path doesn't exist search for it relative to loc (or this file)."""
    if os.path.isfile(path) or path.startswith("$") or validate_url(path):
        return path
    if path.startswith("http"):
        return path
    if path.startswith("hub:"):
        return path
    if loc is None:
        loc = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(loc, path)


def _infer_type_or_str(x):
    try:
        return str2bool(x)
    except:
        try:
            return int(x)
        except ValueError:
            try:
                return float(x)
            except ValueError:
                return x


@export
def parse_extra_args(base_args, extra_args):
    """Parse extra command line arguments based on based names.
    Note:
        special args should be in the form --{base_name}:{special_name}
    :param base_args: List[str], A list of base argument names.
    :param extra_args: List[str], A list of special arguments and values.
    :returns:
        dict, The parsed special settings in the form
        {
            "base": {
                "special": val,
                ...
            },
            ...
        }
    """
    found_args = []
    for arg in base_args:
        key = "{}:".format(arg)
        for extra_arg in extra_args:
            if key in extra_arg:
                found_args.append(extra_arg)
    parser = argparse.ArgumentParser()
    for key in found_args:
        parser.add_argument(key, type=_infer_type_or_str)
    args = parser.parse_known_args(extra_args)[0]
    settings = {arg: {} for arg in base_args}
    args = vars(args)
    for key in args:
        base, extra = key.split(":")
        settings[base][extra] = args[key]
    return settings


@export
def order_json(data, sort_fn: Callable[[List[str]], List[str]] = sorted):
    """Sort json to a consistent order.

    When you hash json that has the some content but is different orders you get
    different fingerprints.

    .. code:: python
        >>> hashlib.sha1(json.dumps({'a': 12, 'b':14}).encode('utf-8')).hexdigest()
        ... '647aa7508f72ece3f8b9df986a206d95fd9a2caf'
        >>> hashlib.sha1(json.dumps({'b': 14, 'a':12}).encode('utf-8')).hexdigest()
        ... 'a22215982dc0e53617be08de7ba9f1a80d232b23'

    Note:
        According to json.org `An object is an unordered set of name/value pairs`. This
        means that the maps (dicts in python) should not contain semantic meaning in the
        order of the key value pairs. This means we are allowed to sort these maps without
        breaking something a user is doing in their custom config

        Similarly `An array is an ordered collection of values` this means that we should
        not sort the lists because that might break the way that a user is using a custom
        list in the json.

    :param data: The json data.
    :param sort_fn: A function that sorts the keys of the dictionary.
    :returns:
        collections.OrderedDict: The data in a consistent order (keys sorted alphabetically).
    """
    new = OrderedDict()
    for key in sort_fn(data.keys()):
        value = data[key]
        # If the value is another map recursively sort that
        if isinstance(value, dict):
            value = order_json(value)
        # If the value is a list, recursively sort any maps that are in it.
        elif isinstance(value, list):
            value = [order_json(v) if isinstance(v, dict) else v for v in value]
        new[key] = value
    return new


KEYS = {
    ('conll_output',),
    ('visdom',),
    ('visdom_name',),
    ('model', 'gpus'),
    ('test_thresh',),
    ('reporting',),
    ('num_valid_to_show',),
    ('train', 'verbose'),
    ('train', 'model_base'),
    ('train', 'model_zip'),
    ('train', 'nsteps'),
    ('test_batchsz',),
    ('basedir',),
}


@export
def remove_extra_keys(config, keys=KEYS):
    """Remove config items that don't effect the model.
    We base most things off of the sha1 hash of the model configs but there
    is a problem. Some things in the config file don't effect the model such
    as the name of the `conll_output` file or if you are using `visdom`
    reporting. This strips out these kind of things so that as long as the model
    parameters match the sha1 will too.
    :param config: dict, The json data.
    :param keys: Set[Tuple[str]], The keys to remove.
    :returns:
        dict, The config with certain keys removed.
    """
    c = deepcopy(config)
    for key in keys:
        x = c
        for k in key[:-1]:
            x = x.get(k)
            if x is None:
                break
        else:
            _ = x.pop(key[-1], None)
    return c


UNORDERED_LIST_KEYS = {
    ('modules',),
}


@export
def sort_list_keys(config, sort_fn=sorted, keys=UNORDERED_LIST_KEYS):
    """Sort the list values associated with given keys.

    Note:
        According to json.org, `An array is an ordered collection of values`.
        This means that we/a user is allowed to encode a semantic meaning to
        the order of a list in the config.

        We personally have several unordered lists in our configs, notably
        the modules list. This function lets us sort those kind of configs
        so that we get a consistent hash.

    :param config: The configuration
    :param sort_fn: The sorting function to call on the value
    :param keys: The keys that should be sorted

    :returns: A copy of the config where all the values associated with
        keys are sorted.
    """
    c = deepcopy(config)
    for key in keys:
        x = c
        for k in key[:-1]:
            x = x.get(k)
            if x is None:
                break
        else:
            if key[-1] in x:
                x[key[-1]] = sort_fn(x[key[-1]])
    return c


@export
def hash_config(config):
    """Hash a json config with sha1.

    :param config: dict, The config to hash.
    :returns:
        str, The sha1 hash.
    """
    # Remove keys that not relevant to reproducing the model itself.
    stripped_config = remove_extra_keys(config)
    # Sort the unordered maps of the configs based on the lexical values of the keys.
    sorted_config = order_json(stripped_config)
    # Sort the list values of specific keys.
    ordered_list_config = sort_list_keys(sorted_config)
    # Convert the config into a json serialized string
    json_bytes = json.dumps(ordered_list_config).encode('utf-8')
    # Hash the resulting config.
    return hashlib.sha1(json_bytes).hexdigest()


def _listdir(model_dir):
    try:
        return os.listdir(model_dir)
    except OSError:
        return []


@export
def find_model_version(model_dir):
    """Find the next usable model version when exporting.

    :param model_dir: `str` The directory we are exporting to.

    :returns: `str` The model version.
    """
    return str(max(chain([0], map(int, filter(lambda x: x.isdigit(), _listdir(model_dir))))) + 1)


@export
def get_output_paths(
        output_dir: str,
        project: Optional[str] = None,
        name: Optional[str] = None,
        version: Optional[str] = None,
        remote: bool = False,
        make_server: bool = True,
        use_version: bool = True,
    ):
    """Create the output paths for export.

    if remote == True:
        f"{output_dir}/client/{project}/{name}/{version}"
        f"{output_dir}/server/{project}/{name}/{version}/model.onnx"
    else:
        f"{output_dir}/{project}/{name}/{version}"

    If either project or name is None then they are skipped in the path.

    If you want to create output that looks the same as the old versions
    then set both project and name to None. This will skip them if remote is
    false or it will add the basename of output_dir to the path if remote.

    :param output_dir: `str` The base of these paths.
    :param project: `str` The first subdir in the created path.
    :param name: `str` The second subdir in the created path.
    :param version: `str` The model version.
    :param remote: `bool` Should we create separate client and server bundles?
    :param make_server: `bool` When false don't actually make the directory that the
        server part will go in. This is because TF really wants to make it itself.
    :param use_version: Should we use
    :returns: `Tuple[str, str]` The client and server output dirs.
    """
    # In this case we use the basename to simulate the old behavior.
    if remote and project is None and name is None:
        project = os.path.basename(output_dir)
    project = project if project is not None else ''
    name = name if name is not None else ''
    client = 'client' if remote else ''
    server = 'server' if remote else ''
    server_path = [output_dir, server, project, name]
    client_path = [output_dir, client, project, name]
    # Sniff the dir and see what version we should use
    if use_version:
        version = find_model_version(os.path.join(*server_path)) if version is None else version
        server_path.append(version)
        client_path.append(version)
    server_path = os.path.join(*server_path)
    client_path = os.path.join(*client_path)
    if remote:
        os.makedirs(client_path)
    if make_server:
        os.makedirs(server_path)
    return client_path, server_path


@export
def get_export_params(
        config: Dict,
        output_dir: Optional[str] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        model_version: Optional[str] = None,
        exporter_type: Optional[str] = None,
        return_labels: Optional[bool] = None,
        is_remote: Optional[bool] = None
):
    """Combine export parameters from the config file and cli arguments.

    :param config: The export block of the config.
    :param output_dir: The base of export paths. (defaults to './models')
    :param project: The name of the project this model is for.
    :param name: The name of this model (often the use case for it, `ner`, `intent` etc).
    :param model_version: The version of this model.
    :param exporter_type: The name of the exporter to use (defaults to 'default')
    :param return_labels: Should labels be returned? (defaults to False)
    :param is_remote: Should the bundle be split into client and server dirs.
    :param user_version

    :returns: `Tuple[str, str, str, str, str, bool, bool]`
        The output_dir, project, name, model_version, exporter_type, return_labels, and remote
    """
    project = project if project is not None else config.get('project')
    name = name if name is not None else config.get('name')
    output_dir = output_dir if output_dir is not None else config.get('output_dir', './models')
    output_dir = os.path.expanduser(output_dir)
    model_version = model_version if model_version is not None else config.get('model_version')
    exporter_type = exporter_type if exporter_type is not None else config.get('type', config.get('exporter_type', 'default'))
    return_labels = return_labels if return_labels is not None else config.get('return_labels', False)
    return_labels = str2bool(return_labels)
    is_remote = is_remote if is_remote is not None else config.get('is_remote', True)
    is_remote = str2bool(is_remote)
    return output_dir, project, name, model_version, exporter_type, return_labels, is_remote


def create_metadata(inputs, outputs, sig_name, model_name, lengths_key=None, beam=None, return_labels=False, preproc='client'):
    meta = {
        'inputs': inputs,
        'outputs': outputs,
        'signature_name': sig_name,
        'metadata': {
            'exported_model': str(model_name),
            'exported_time': str(datetime.utcnow()),
            'return_labels': return_labels,
            'preproc': preproc,
        }
    }
    if lengths_key:
        meta['lengths_key'] = lengths_key
    if beam:
        meta['beam'] = beam

    return meta


def save_to_bundle(output_path, directory, assets=None, zip_results=False):
    """Save files to the exported bundle.

    :vocabs
    :vectorizers
    :labels
    :assets
    :output_path  the bundle output_path. vocabs, vectorizers know how to save themselves.
    """
    for filename in os.listdir(directory):
        if filename.startswith('vocabs') or \
           filename.endswith(".labels") or \
           filename.startswith('vectorizers'):
            shutil.copy(os.path.join(directory, filename), os.path.join(output_path, filename))

    if assets:
        asset_file = os.path.join(output_path, 'model.assets')
        write_json(assets, asset_file)

    if zip_results:
        zip_files(output_path, False)
        delete_old_copy(output_path)

def create_feature_exporter_field_map(feature_section, default_exporter_field='tokens'):
    feature_exporter_field_map = {}
    for feature_desc in feature_section:
        if feature_desc.get('exporter_field') is None:
            feature_exporter_field_map[feature_desc['name']] = default_exporter_field
        else:
            feature_exporter_field_map[feature_desc['name']] = feature_desc['exporter_field']
    return feature_exporter_field_map

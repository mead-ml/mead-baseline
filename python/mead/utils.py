import os
import json
import hashlib
import argparse
from copy import deepcopy
from collections import OrderedDict
from baseline.utils import export, str2bool, read_config_file

__all__ = []
exporter = export(__all__)


@exporter
def print_dataset_info(dataset):
    print("[train file]: {}".format(dataset['train_file']))
    print("[valid file]: {}".format(dataset['valid_file']))
    print("[test file]: {}".format(dataset['test_file']))
    vocab_file = dataset.get('vocab_file')
    if vocab_file is not None:
        print("[vocab file]: {}".format(vocab_file))
    label_file = dataset.get('label_file')
    if label_file is not None:
        print("[label file]: {}".format(label_file))


@exporter
def read_config_file_or_json(config, name=''):
    if isinstance(config, (dict, list)):
        return config
    config = os.path.expanduser(config)
    if os.path.exists(config):
        return read_config_file(config)
    raise Exception('Expected {} config file or a JSON object.'.format(name))


@exporter
def get_mead_settings(mead_settings_config):
    if mead_settings_config is None:
        return {}
    return read_config_file_or_json(mead_settings_config, 'mead settings')


@exporter
def index_by_label(object_list):
    objects = {x['label']: x for x in object_list}
    return objects


@exporter
def convert_path(path, loc=None):
    """If the provided path doesn't exist search for it relative to loc (or this file)."""
    if os.path.isfile(path):
        return path
    if path.startswith("$"):
        return path
    if loc is None:
        loc = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(loc, path)


def _infer_type_or_str(x):
    try:
        return str2bool(x)
    except:
        try:
            return float(x)
        except ValueError:
            return x


@exporter
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


@exporter
def order_json(data):
    """Sort json to a consistent order.
    When you hash json that has the some content but is different orders you get
    different fingerprints.
    In:  hashlib.sha1(json.dumps({'a': 12, 'b':14}).encode('utf-8')).hexdigest()
    Out: '647aa7508f72ece3f8b9df986a206d95fd9a2caf'
    In:  hashlib.sha1(json.dumps({'b': 14, 'a':12}).encode('utf-8')).hexdigest()
    Out: 'a22215982dc0e53617be08de7ba9f1a80d232b23'
    This function sorts json by key so that hashes are consistent.
    Note:
        In our configs we only have lists where the order doesn't matter so we
        can sort them for consistency. This would have to change if we add a
        config field that needs order we will need to refactor this.
    :param data: dict, The json data.
    :returns:
        collections.OrderedDict: The data in a consistent order (keys sorted alphabetically).
    """
    new = OrderedDict()
    for (key, value) in sorted(data.items(), key=lambda x: x[0]):
        if isinstance(value, dict):
            value = order_json(value)
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
    ('test_batchsz'),
    ('basedir'),
}


@exporter
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


@exporter
def hash_config(config):
    """Hash a json config with sha1.
    :param config: dict, The config to hash.
    :returns:
        str, The sha1 hash.
    """
    stripped_config = remove_extra_keys(config)
    sorted_config = order_json(stripped_config)
    json_bytes = json.dumps(sorted_config).encode('utf-8')
    return hashlib.sha1(json_bytes).hexdigest()

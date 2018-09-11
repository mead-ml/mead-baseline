import os
import json
from baseline.utils import export, str2bool
from mead.mime_type import mime_type
import hashlib
import zipfile
import argparse

__all__ = []
exporter = export(__all__)


@exporter
def index_by_label(dataset_file):
    with open(dataset_file) as f:
        datasets_list = json.load(f)
        datasets = dict((x["label"], x) for x in datasets_list)
        return datasets


@exporter
def convert_path(path, loc=None):
    """If the provided path doesn't exist search for it relative to loc (or this file)."""
    if os.path.isfile(path):
        return path
    if loc is None:
        loc = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(loc, path)


@exporter
def _infer_type_or_str(x):
    try:
        return str2bool(x)
    except:
        try:
            return float(x)
        except ValueError:
            return x

@exporter
def modify_reporting_hook_settings(reporting_settings, reporting_args_mead, reporting_hooks):
    reporting_arg_keys = []
    for x in reporting_hooks:
        for var in reporting_args_mead:
            if "{}:".format(x) in var:
                reporting_arg_keys.append(var)
    parser = argparse.ArgumentParser()
    for key in reporting_arg_keys:
        parser.add_argument(key, type=_infer_type_or_str)
    args = parser.parse_known_args()[0]
    for key in vars(args):
        this_hook, var = key.split(":")
        reporting_settings[this_hook].update({var: vars(args)[key]})



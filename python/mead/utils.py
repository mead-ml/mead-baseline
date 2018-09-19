import os
import json
from baseline.utils import export

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


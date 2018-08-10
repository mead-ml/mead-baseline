import os
import json
from baseline.utils import export
from mead.mime_type import mime_type
import hashlib
import zipfile

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
def unzip_model(path):
    """If the path for a model file is a zip file, unzip it in /tmp and return the unzipped path"""
    if not os.path.exists(path) or not mime_type(path) == "application/zip":
        return path
    with open(path, 'rb') as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()
    temp_dir = os.path.join("/tmp/", sha1)
    if not os.path.exists(temp_dir):
        print("unzipping model before exporting")
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
    temp_dir = os.path.join(temp_dir, os.listdir(temp_dir)[0])
    path = os.path.join(temp_dir, [x[:-6] for x in os.listdir(temp_dir) if 'index' in x][0])
    return path

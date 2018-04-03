import json


def index_by_label(dataset_file):
    with open(dataset_file) as f:
        datasets_list = json.load(f)
        datasets = dict((x["label"], x) for x in datasets_list)
        return datasets

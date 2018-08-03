import json
import pymongo
import pandas as pd
import os
from collections import OrderedDict

__all__ = ["log2json", "order_json"]


def log2json(log_file):
    s = []
    with open(log_file) as f:
        for line in f:
            x = line.replace("'", '"')
            s.append(json.loads(x))
    return s


def order_json(j):
    new = OrderedDict()
    for key in sorted(j.keys()):
        if isinstance(j[key], dict):
            value = order_json(j[key])
        elif isinstance(j[key], list):
            value = sorted(j[key])
        else:
            value = j[key]
        new[key] = value
    return new

import json
import pymongo
import pandas as pd
import os
from collections import OrderedDict
import numpy as np

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


def sort_ascending(metric):
    return metric == "avg_loss" or metric == "perplexity"


def df_summary_exp(df):
    return df.groupby("sha1").agg([len, np.mean, np.std, np.min, np.max]) \
        .rename(columns={'len': 'num_exps', 'amean': 'mean', 'amin': 'min', 'amax': 'max'})


def df_summary_task(df, metric=None, sort=None):
    result_frame = df.groupby("sha1").agg([len, np.mean, np.std, np.min, np.max])\
        .rename(columns={'len': 'num_exps', 'amean': 'mean', 'amin': 'min', 'amax': 'max'})
    metrics = list(metric)
    if len(metric) == 1:
        return result_frame.sort_values([(metrics[0], 'mean')], ascending=sort_ascending(metric))
    if sort:
        return result_frame.sort_values([(sort, 'mean')], ascending=sort_ascending(metric))
    return result_frame

import json
import pandas as pd
from collections import OrderedDict
import numpy as np
from baseline.utils import listify

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


def df_get_results(result_frame, dataset, num_exps, metric, sort):
    datasets = result_frame.dataset.unique()
    if dataset not in datasets:
        return None
    dsr = result_frame[result_frame.dataset == dataset]
    if dsr.empty:
        return None
    df = pd.DataFrame()
    if num_exps is not None:
        for gname, rframe in result_frame.groupby("sha1"):
            rframe = rframe.copy()
            rframe['date'] =pd.to_datetime(rframe.date)
            rframe = rframe.sort_values(by='date', ascending=False).head(int(num_exps))
            df = df.append(rframe)
        result_frame = df
    result_frame = result_frame.drop(columns=["id"])
    result_frame = result_frame.groupby("sha1").agg([len, np.mean, np.std, np.min, np.max])\
        .rename(columns={'len': 'num_exps', 'amean': 'mean', 'amin': 'min', 'amax': 'max'})
    metrics = listify(metric)
    if len(metrics) == 1:
        result_frame = result_frame.sort_values([(metrics[0], 'mean')], ascending=sort_ascending(metric))
    if sort:
        result_frame = result_frame.sort_values([(sort, 'mean')], ascending=sort_ascending(metric))
    if result_frame.empty:
        return None
    return result_frame


def df_experimental_details(result_frame, sha1, users, sort, metric, num_exps):
    result_frame = result_frame[result_frame.sha1 == sha1]
    if result_frame.empty:
        return None
    if num_exps is not None:
        result_frame = result_frame.copy()
        result_frame['date'] =pd.to_datetime(result_frame.date)
        result_frame = result_frame.sort_values(by='date', ascending=False).head(int(num_exps))
    if users is not None:
        df = pd.DataFrame()
        for user in users:
            df = df.append(result_frame[result_frame.username == user])
        result_frame = result_frame
    metrics = list(metric)
    if len(metrics) == 1:
        result_frame = result_frame.sort_values([metrics[0]], ascending=sort_ascending(metric))
    if sort:
        result_frame = result_frame.sort_values([sort], ascending=sort_ascending(metric))
    if result_frame.empty:
        return None
    return result_frame

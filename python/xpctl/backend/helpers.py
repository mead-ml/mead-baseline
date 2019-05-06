import json
import pandas as pd
from collections import OrderedDict
import numpy as np
from baseline.utils import listify, export
#from xpctl.dto import MongoResultSet, MongoResult

__all__ = []
exporter = export(__all__)

METRICS_SORT_ASCENDING = ['avg_loss', 'perplexity']


@exporter
def log2json(log_file):
    s = []
    with open(log_file) as f:
        for line in f:
            x = line.replace("'", '"')
            s.append(json.loads(x))
    return s


def sort_ascending(metric):
    return metric == "avg_loss" or metric == "perplexity"


def df_summary_exp(df):
    return df.groupby("sha1").agg([len, np.mean, np.std, np.min, np.max]) \
        .rename(columns={'len': 'num_exps', 'amean': 'mean', 'amin': 'min', 'amax': 'max'})


def df_get_results(result_frame, dataset, num_exps, num_exps_per_config, metric, sort):
    datasets = result_frame.dataset.unique()
    if dataset not in datasets:
        return None
    dsr = result_frame[result_frame.dataset == dataset]
    if dsr.empty:
        return None
    df = pd.DataFrame()
    if num_exps_per_config is not None:
        for gname, rframe in result_frame.groupby("sha1"):
            rframe = rframe.copy()
            rframe['date'] =pd.to_datetime(rframe.date)
            rframe = rframe.sort_values(by='date', ascending=False).head(int(num_exps_per_config))
            df = df.append(rframe)
        result_frame = df

    result_frame = result_frame.drop(["id"], axis=1)
    result_frame = result_frame.groupby("sha1").agg([len, np.mean, np.std, np.min, np.max])\
        .rename(columns={'len': 'num_exps', 'amean': 'mean', 'amin': 'min', 'amax': 'max'})
    metrics = listify(metric)
    if len(metrics) == 1:
        result_frame = result_frame.sort_values([(metrics[0], 'mean')], ascending=sort_ascending(metric))
    if sort:
        result_frame = result_frame.sort_values([(sort, 'mean')], ascending=sort_ascending(metric))
    if result_frame.empty:
        return None
    if num_exps is not None:
        result_frame = result_frame.head(num_exps)
    return result_frame


def df_experimental_details(result_frame, sha1, users, sort, metric, num_exps):
    result_frame = result_frame[result_frame.sha1 == sha1]
    if result_frame.empty:
        return None
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
    if num_exps is not None:
        result_frame = result_frame.head(num_exps)
    return result_frame


def get_experiment_label(config_obj, task, **kwargs):
    if kwargs.get('label', None) is not None:
        return kwargs['label']
    if 'description' in config_obj:
        return config_obj['description']
    else:
        model_type = config_obj.get('model_type', 'default')
        backend = config_obj.get('backend', 'tensorflow')
        return "{}-{}-{}".format(task, backend, model_type)


def aggregate_results(resultset, groupby_key, num_exps_per_reduction, num_exps):
    grouped_result = resultset.groupby(groupby_key)
    
    aggregate_fns = {'min': np.min, 'max': np.max, 'avg': np.mean, 'std': np.std}
    
    return grouped_result.reduce(aggregate_fns=aggregate_fns)



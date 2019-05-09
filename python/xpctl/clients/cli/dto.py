import pandas as pd
from swagger_client.models import Experiment, ExperimentAggregate, Result, AggregateResult, TaskSummary
from typing import List
from baseline.utils import write_config_file, read_config_file
import json


def pack_result(results: List[Result]):
    """ List of results to event data"""
    metrics = set([r.metric for r in results])
    d = {metric: [] for metric in metrics}
    for result in results:
        d[result.metric].append(result.value)
    return pd.DataFrame(d)


def pack_aggregate_result(results: List[AggregateResult]):
    metrics = [r.metric for r in results]
    metrics = set(metrics)
    d = {metric: {} for metric in metrics}
    for result in results:
        for value in result.values:
            aggregate_fn = value.aggregate_fn
            score = value.score
            if aggregate_fn in d[result.metric]:
                d[result.metric][aggregate_fn].append(score)
            else:
                d[result.metric][aggregate_fn] = [score]

    dfs = {metric: pd.DataFrame.from_dict(d[metric]) for metric in metrics}
    return pd.concat(dfs.values(), axis=1, keys=dfs.keys())


def get_prop_value(exp, prop_name):
    return exp.__dict__.get('_'+prop_name)


def insert_in_df(prop_name_loc, df, exp):
    for prop_name, location in prop_name_loc.items():
        df.insert(location, column=prop_name, value=[get_prop_value(exp, prop_name)]*len(df))
        
        
def experiment_to_df(exp: Experiment, prop_name_loc={}, event_type='test_events', sort=None):
    prop_name_loc = {'sha1': 0, 'id': 1, 'username':  2} if not prop_name_loc else prop_name_loc
    if event_type == 'train_events' and exp.train_events:
        result_df = pack_result(exp.train_events)
        insert_in_df(prop_name_loc, result_df, exp)
    if event_type == 'valid_events' and exp.valid_events:
        result_df = pack_result(exp.valid_events)
        insert_in_df(prop_name_loc, result_df, exp)
    if event_type == 'test_events' and exp.test_events:
        result_df = pack_result(exp.test_events)
        insert_in_df(prop_name_loc, result_df, exp)
    if sort is not None:
        result_df.sort_values(by=sort, inplace=True)
    return result_df


def experiment_aggregate_to_df(exp_agg: ExperimentAggregate, prop_name_loc, event_type='test_events'):
    event_dfs = []
    if event_type == 'train_events':
        train_df = pack_aggregate_result(exp_agg.train_events)
        insert_in_df(prop_name_loc, train_df, exp_agg)
        event_dfs.append(train_df)
    if event_type == 'valid_events':
        valid_df = pack_aggregate_result(exp_agg.valid_events)
        insert_in_df(prop_name_loc, valid_df, exp_agg)
        event_dfs.append(valid_df)
    if event_type == 'test_events':
        test_df = pack_aggregate_result(exp_agg.test_events)
        insert_in_df(prop_name_loc, test_df, exp_agg)
        event_dfs.append(test_df)
    result_df = pd.DataFrame()
    for event_df in event_dfs:
        result_df = result_df.append(event_df)
    return result_df


def experiment_aggregate_list_to_df(exp_aggs: List[ExperimentAggregate], event_type='test_events'):
    result_df = pd.DataFrame()
    prop_name_loc = {'sha1': 0, 'num_exps': 1}
    for exp_agg in exp_aggs:
        result_df = result_df.append(experiment_aggregate_to_df(exp_agg, prop_name_loc, event_type))
    return result_df


def experiment_list_to_df(exps: List[Experiment], prop_name_loc={}, event_type='test_events'):
    result_df = pd.DataFrame()
    prop_name_loc = {'sha1': 0, 'id': 1, 'username':  2} if not prop_name_loc else prop_name_loc
    for exp in exps:
        result_df = result_df.append(experiment_to_df(exp, prop_name_loc, event_type, sort=None))
    return result_df


def write_to_config_file(config_obj, filename):
    write_config_file(config_obj, filename)


def task_summary_to_df(tasksummary: TaskSummary):
    def identity(x): return x
    summary = tasksummary.summary
    all_results = []
    for dataset in summary:
        for user, num_exps in summary[dataset]:
            all_results.append([user, dataset, num_exps])
    return pd.DataFrame(all_results, columns=['user', 'dataset', 'num_exps']).groupby(['user', 'dataset'])\
        .agg([identity]).rename(columns={'identity': ''})


def task_summaries_to_df(tasksummaries: List[TaskSummary]):
    def identity(x): return x
    all_results = []
    for tasksummary in tasksummaries:
        task = tasksummary.task
        summary = tasksummary.summary
        for dataset in summary:
            for user, num_exps in summary[dataset]:
                all_results.append([task, user, dataset, num_exps])
    return pd.DataFrame(all_results, columns=['task', 'user', 'dataset', 'num_exps']).groupby(['task', 'user', 'dataset'])\
        .agg([identity]).rename(columns={'identity': ''})


def read_logs(file_name):
    logs = []
    with open(file_name) as f:
        for line in f:
            logs.append(json.loads(line))
    return logs


def convert_to_result(event):
    results = []
    non_metrics = ['tick_type', 'tick', 'phase']
    metrics = event.keys() - non_metrics
    for metric in metrics:
        results.append(Result(
             metric=metric,
             value=event[metric],
             tick_type=event['tick_type'],
             tick=event['tick'],
             phase=event['phase']
            )
        )
    return results
    

def flatten(_list):
    return [item for sublist in _list for item in sublist]


def to_experiment(task, configf, logf, user, label):
    events_obj = read_logs(logf)
    train_events = flatten(
        [convert_to_result(event) for event in list(filter(lambda x: x['phase'] == 'Train', events_obj))]
    )
    valid_events = flatten(
        [convert_to_result(event) for event in list(filter(lambda x: x['phase'] == 'Valid', events_obj))]
    )
    test_events = flatten(
        [convert_to_result(event) for event in list(filter(lambda x: x['phase'] == 'Test', events_obj))]
    )
    config = json.dumps(read_config_file(configf))
    return Experiment(
        task=task,
        username=user,
        label=label,
        train_events=train_events,
        valid_events=valid_events,
        test_events=test_events,
        config=config
    )

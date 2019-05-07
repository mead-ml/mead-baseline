import pandas as pd
from swagger_client.models import Experiment, Result
from typing import List


def pack_event(results: List[Result]):
    """ List of results to event data"""
    metrics = set([r.metric for r in results])
    d = {metric: [] for metric in metrics}
    for result in results:
        d[result.metric].append(result.value)
    return pd.DataFrame(d)


def get_prop_value(exp, prop_name):
    return exp.__dict__.get('_'+prop_name)


def insert_in_df(prop_name_loc, df, exp):
    for prop_name, location in prop_name_loc.items():
        df.insert(location, column=prop_name, value=[get_prop_value(exp, prop_name)]*len(df))
        
        
def experiment_to_data_frame(exp: Experiment, prop_name_loc={}, sort=None):
    event_dfs = []
    prop_name_loc = {'sha1': 0, 'id': 1, 'user':  2} if not prop_name_loc else prop_name_loc
    if exp.train_events:
        train_df = pack_event(exp.train_events)
        insert_in_df(prop_name_loc, train_df, exp)
        event_dfs.append(train_df)
    if exp.valid_events:
        valid_df = pack_event(exp.valid_events)
        insert_in_df(prop_name_loc, valid_df, exp)
        event_dfs.append(valid_df)
    if exp.test_events:
        test_df = pack_event(exp.test_events)
        insert_in_df(prop_name_loc, test_df, exp)
        event_dfs.append(test_df)
    result_df = pd.DataFrame()
    for event_df in event_dfs:
        result_df = result_df.append(event_df)
    if sort is not None:
        result_df.sort_values(by=sort, inplace=True)
    return result_df


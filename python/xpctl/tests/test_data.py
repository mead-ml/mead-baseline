import pytest
import json
import os
from xpctl.mongo.backend import MongoRepo
import pickle
import numpy as np
from pprint import pprint
from xpctl.tests.assert_almost import assert_dict_almost_equal


@pytest.fixture
def setup():
    creds = json.load(open(os.path.expanduser(json.load(open('xpctlcred-loc.json'))['loc'])))
    return creds


def test_resultset_mongo(setup):
    creds = setup
    input_ = json.load(open('data/mongosample1.json'))
    m = MongoRepo(**creds)
    rs = m.mongo_to_resultset(input_, event_type='test_events', metrics=[])
    rs_stored = pickle.load(open('data/mongosample1.resultset.pickle', 'rb'))
    assert len(rs) == len(rs_stored)
    matched = 0
    for d1 in rs:
        for d2 in rs_stored:
            if d1.__dict__ == d2.__dict__:
                matched += 1
    assert len(rs) == matched


def _reduction_dim(resultset, reduction_dim, expected_result, aggregate_fns):
    rgs = resultset.groupby(reduction_dim)
    agg_results = rgs.reduce(aggregate_fns)
    results = {}
    for x in agg_results:
        if x.get_prop(reduction_dim) not in results:
            results[x.get_prop(reduction_dim)] = {}
        results[x.get_prop(reduction_dim)][x.metric] = x.values
    return assert_dict_almost_equal(expected_result, results) is None
    

def _test_reduction():
    expected_results = json.load(open('data/results-mongosample1.json'))
    resultset = pickle.load(open('data/mongosample1.resultset.pickle', 'rb'))
    aggregate_fns = {'mean': np.mean, 'std': np.std}
    reduction_dims = ['sha1', '_id']
    for reduction_dim in reduction_dims:
        assert _reduction_dim(resultset, reduction_dim, expected_results[reduction_dim], aggregate_fns)


def test_reduction_mongo(setup):
    creds = setup
    input_ = json.load(open('data/mongosample2.json'))
    m = MongoRepo(**creds)
    resultset = m.mongo_to_resultset(input_, event_type='train_events', metrics=[])
    expected_results = json.load(open('data/results-mongosample2.json'))
    aggregate_fns = {'mean': np.mean, 'std': np.std}
    reduction_dims = ['sha1']
    for reduction_dim in reduction_dims:
        assert _reduction_dim(resultset, reduction_dim, expected_results[reduction_dim], aggregate_fns)


def test_reduction_mongo_2(setup):
    creds = setup
    input_ = json.load(open('data/mongosample3.json'))
    m = MongoRepo(**creds)
    event_types = ['train_events', 'dev_events', 'test_events']
    for event_type in event_types:
        resultset = m.mongo_to_resultset(input_, event_type=event_type, metrics=[])
        expected_results = json.load(open('data/results-mongosample3-{}.json'.format(event_type.replace('_', '-'))))
        aggregate_fns = {'mean': np.mean, 'std': np.std}
        reduction_dims = ['sha1']
        for reduction_dim in reduction_dims:
            assert _reduction_dim(resultset, reduction_dim, expected_results[reduction_dim], aggregate_fns)

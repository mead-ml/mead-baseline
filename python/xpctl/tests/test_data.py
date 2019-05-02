import pickle
import numpy as np
from xpctl.tests.assert_almost import assert_dict_almost_equal
from pprint import pprint


def test_reduction_one_dim():
    resultset = pickle.load(open('data/mongosample3.resultset.pickle', 'rb'))
    rgs = resultset.groupby('sha1')
    aggregate_fns = {'mean': np.mean, 'std': np.std}
    agg_results = rgs.reduce(aggregate_fns, event_type='valid_events')
    pprint([x.__dict__ for x in agg_results])


def _reduction_dim(resultset, reduction_dim, expected_result, aggregate_fns):
    rgs = resultset.groupby(reduction_dim)
    agg_results = rgs.reduce(aggregate_fns)
    results = {}
    for x in agg_results:
        if x.get_prop(reduction_dim) not in results:
            results[x.get_prop(reduction_dim)] = {}
        results[x.get_prop(reduction_dim)][x.metric] = x.values
    return assert_dict_almost_equal(expected_result, results) is None


# def test_reduction():
#     expected_results = json.load(open('data/results-mongosample1.json'))
#     resultset = pickle.load(open('data/mongosample1.resultset.pickle', 'rb'))
#     aggregate_fns = {'mean': np.mean, 'std': np.std}
#     reduction_dims = ['sha1', '_id']
#     for reduction_dim in reduction_dims:
#         assert _reduction_dim(resultset, reduction_dim, expected_results[reduction_dim], aggregate_fns)
#

# def _test_reduction_mongo(setup):
#     creds = setup
#     input_ = json.load(open('data/mongosample2.json'))
#     m = MongoRepo(**creds)
#     resultset = m.mongo_to_resultset(input_, event_type='train_events', metrics=[])
#     expected_results = json.load(open('data/results-mongosample2.json'))
#     aggregate_fns = {'mean': np.mean, 'std': np.std}
#     reduction_dims = ['sha1']
#     for reduction_dim in reduction_dims:
#         assert _reduction_dim(resultset, reduction_dim, expected_results[reduction_dim], aggregate_fns)
#
#
# def _test_reduction_mongo_2(setup):
#     creds = setup
#     input_ = json.load(open('data/mongosample3.json'))
#     m = MongoRepo(**creds)
#     event_types = ['train_events', 'dev_events', 'test_events']
#     for event_type in event_types:
#         resultset = m.mongo_to_resultset(input_, event_type=event_type, metrics=[])
#         expected_results = json.load(open('data/results-mongosample3-{}.json'.format(event_type.replace('_', '-'))))
#         aggregate_fns = {'mean': np.mean, 'std': np.std}
#         reduction_dims = ['sha1']
#         for reduction_dim in reduction_dims:
#             assert _reduction_dim(resultset, reduction_dim, expected_results[reduction_dim], aggregate_fns)
#
#
# def _test_resultsetgroup_trim():
#     resultset = pickle.load(open('data/mongosample1.resultset.pickle', 'rb'))
#     rgs = resultset.groupby('sha1')
#     expected = {
#         '1a870728e04470a07643c5d9cff33329c004751f': 5,
#         '67105e2108885c5ee08e211537fbda602f2ba254': 5,
#         '8ab6ab6ee8fdf14b111223e2edf48750c30c7e51': 15
#     }
#     for sha1, resultset in rgs.grouped_resultsets.items():
#         assert expected[sha1] == len(resultset)
#
#     multiple_exps = rgs.get('8ab6ab6ee8fdf14b111223e2edf48750c30c7e51')
#     grouped = multiple_exps.groupby('_id')
#     assert len(grouped) == 3
#     grouped.trim(2)
#     assert len(grouped) == 2
#     grouped.trim(1)
#     assert len(grouped) == 1

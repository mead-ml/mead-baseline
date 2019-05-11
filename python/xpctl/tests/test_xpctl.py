import json
import pytest
from swagger_client import Configuration
from swagger_client.api import XpctlApi
from swagger_client import ApiClient
from swagger_client.models import Experiment, Result
from swagger_client.rest import ApiException
from mead.utils import hash_config
from collections import namedtuple
import datetime
import random
import string
from functools import wraps

TASK = 'test_xpctl'
EIDS = []
HOST = 'localhost:5310/v2'
API = None


def clean_db(func):
    @wraps(func)
    def clean(*args, **kwargs):
        try:
            eids = [x.eid for x in API.list_experiments_by_prop(TASK, prop=None, value=None)]
            for eid in eids:
                response = API.remove_experiment(TASK, eid)
                if response.response_type == 'error':
                    raise RuntimeError('can not clean db')
        except ApiException:
            # there's probably nothing in db
            return func(*args, **kwargs)
        return func(*args, **kwargs)
    return clean


@pytest.fixture(scope='session', autouse=True)
def setup():
    config = Configuration(host=HOST)
    api_client = ApiClient(config)
    global API
    API = XpctlApi(api_client)


def _put_one_exp(exp):
    result = API.put_result(TASK, exp)
    if result.response_type == 'error':
        raise RuntimeError(result.message)
    else:
        global EIDS
        EIDS.append(result.message)
        return result.message


@clean_db
def test_put_result_blank(setup):
    """ you can not insert an experiment with no results"""
    exp = Experiment(
        config='{"test": "test"}',
        train_events=[],
        valid_events=[],
        test_events=[]
    )
    try:
        _ = _put_one_exp(exp)
    except ApiException:
        return True


@clean_db
def test_put_result_one(setup):
    """ test a proper insertion"""
    exp = Experiment(
        config='{"test": "test"}',
        train_events=[],
        valid_events=[],
        test_events=[Result(metric='t', tick_type='t', phase='Test', tick=0, value=0.1)]
    )
    eid = _put_one_exp(exp)
    result = API.list_experiments_by_prop(TASK, prop='eid', value=eid)
    assert len(result) == 1


@clean_db
def test_experiment_details(setup):
    date = datetime.datetime.utcnow().isoformat()
    label = 'test_label'
    dataset = 'test_dataset'
    config = '{"test":"test"}'
    username = 'user'
    hostname = 'host'
    exp = Experiment(
        dataset=dataset,
        label=label,
        exp_date=date,
        config=config,
        username=username,
        hostname=hostname,
        train_events=[],
        valid_events=[],
        test_events=[Result(metric='t', tick_type='t', phase='Test', tick=0, value=0.1)]
    )
    eid = _put_one_exp(exp)
    try:
        result = API.list_experiments_by_prop(TASK, prop='eid', value=eid)
    except ApiException:
        print(eid)
        return False
    exp = result[0]
    assert exp.dataset == 'test_dataset'
    assert exp.label == label
    assert (exp.exp_date == date) or (exp.exp_date[:-1] == date)  # sql inserts Z at the end
    assert exp.sha1 == hash_config(json.loads('{"test": "test"}'))
    assert exp.username == username
    assert exp.hostname == hostname


def _generate_random_string(num_chars=20):
    return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
                   for _ in range(num_chars))


def _find_by_prop_unique(prop, expected_value):
    """
    since the property has an unique value, we should see just one experiment with that property
    :param prop: 'dataset', 'username' etc
    :param expected_value: a value generated from _generate_random_string
    :return:
    """
    try:
        result = API.list_experiments_by_prop(TASK, prop=prop, value=expected_value)
        return len(result) == 1
    except ApiException:
        return False


@clean_db
def test_update_property(setup):
    date = datetime.datetime.utcnow().isoformat()
    label = _generate_random_string()
    dataset = _generate_random_string()
    config = '{"test":"test"}'
    username = _generate_random_string()
    hostname = _generate_random_string()
    exp = Experiment(
        dataset='d',
        label='l',
        exp_date=date,
        config=config,
        username='u',
        hostname='h',
        train_events=[],
        valid_events=[],
        test_events=[Result(metric='t', tick_type='t', phase='Test', tick=0, value=0.1)]
    )
    eid = _put_one_exp(exp)
    API.update_property(TASK, eid, 'dataset', dataset)
    API.update_property(TASK, eid, 'label', label)
    API.update_property(TASK, eid, 'username', username)
    API.update_property(TASK, eid, 'hostname', hostname)
    assert _find_by_prop_unique(prop='label', expected_value=label)
    assert _find_by_prop_unique(prop='dataset', expected_value=dataset)
    assert _find_by_prop_unique(prop='username', expected_value=username)
    assert _find_by_prop_unique(prop='hostname', expected_value=hostname)


def _test_list_experiments_by_prop(prop, value, experiments):
    results = API.list_experiments_by_prop(TASK, prop=prop, value=value)
    expected = [exp.eid for exp in experiments if getattr(exp, prop) == value]
    results = [r.eid for r in results]
    return set(expected) == set(results)


@clean_db
def test_list_experiments_by_prop(setup):
    """
    list_experiments_by_prop finds some experiments by a (prop, value) pair. also allows some filtering:
    :param setup:
    :return:
    """
    configs = ['{"c1":"c1"}', '{"c2":"c2"}']
    datasets = ['d1', 'd2']
    labels = ['l1', 'l2']
    users = ['u1', 'u2', 'u3']
    metrics = ['f1', 'acc', 'random']
    test_events = [Result(metric=metric, value=0.5, tick_type='EPOCH', tick=0, phase='Test') for metric in metrics]
    experiments = []
    exp_detail = namedtuple('exp_detail', ['eid', 'sha1', 'dataset', 'label', 'username'])
    for config in configs:
        for dataset in datasets:
            for label in labels:
                for username in users:
                    result = _put_one_exp(
                        Experiment(
                            config=config,
                            dataset=dataset,
                            label=label,
                            username=username,
                            train_events=[],
                            valid_events=[],
                            test_events=test_events
                        )
                    )
                    experiments.append(exp_detail(eid=result, sha1=hash_config(json.loads(config)), dataset=dataset,
                                                  label=label, username=username))

    # find by a property and group by different reduction dims
    for prop, values in {'dataset': datasets, 'label': labels, 'username': users}.items():
        for value in values:
            assert _test_list_experiments_by_prop(prop=prop, value=value, experiments=experiments)

    # test the `users` filter work
    prop = 'dataset'
    value = 'd1'
    users = [['u1', 'u2'], ['u1']]
    for user in users:
        results = API.list_experiments_by_prop(TASK, prop=prop, value=value, user=user)
        result_eids = [x.eid for x in results]
        expected_eids = [x.eid for x in experiments if x.dataset == 'd1' and x.username in user]
        assert set(result_eids) == set(expected_eids)
    # test sort works
    config = configs[0]
    metrics = ['f1', 'acc']
    exp_value = namedtuple('exp_value', ['eid', 'value'])
    experiments = []
    dataset = _generate_random_string()  # generate a unique dataset
    for value in [0.5, 0.7, 0.8]:
        test_events = [Result(metric=metric, value=value, tick_type='EPOCH', tick=0, phase='Test') for metric in metrics]
        result = _put_one_exp(
            Experiment(
                config=config,
                train_events=[],
                valid_events=[],
                test_events=test_events,
                dataset=dataset
            )
        )
        experiments.append(exp_value(eid=result, value=value))
    results = API.list_experiments_by_prop(TASK, prop='dataset', value=dataset, sort='f1')  # get results only from that
    # unique dataset, sort by f1
    _max = experiments[-1]
    assert results[0].eid == _max.eid


def _test_reduction_dim(prop, value, reduction_dim, experiments):
    """
    The results from API should return proper groups
    :param prop:
    :param value:
    :param reduction_dim:
    :param experiments:
    :return:
    """
    results = API.get_results_by_prop(TASK, prop=prop, value=value, reduction_dim=reduction_dim)
    expected = {}
    for item in [getattr(exp, reduction_dim) for exp in experiments if getattr(exp, prop) == value]:
        if item not in expected:
            expected[item] = 1
        else:
            expected[item] += 1
    results = {getattr(r, reduction_dim): r.num_exps for r in results}
    return expected == results


@clean_db
def test_get_results_by_prop(setup):
    """
    get_results_by_prop first finds some experiments by a (prop, value) pair and then groups them by a reduction_dim.
    typically we use prop='dataset' and reduction_dim='sha1', but the API supports more general calls
    :param setup:
    :return:
    """
    configs = ['{"c1":"c1"}', '{"c2":"c2"}']
    datasets = ['d1', 'd1', 'd2']
    labels = ['l1', 'l2', 'l2']
    metrics = ['f1', 'acc', 'random']
    test_events = [Result(metric=metric, value=0.5, tick_type='EPOCH', tick=0, phase='Test') for metric in metrics]
    experiments = []
    exp_detail = namedtuple('exp_detail', ['eid', 'sha1', 'dataset', 'label'])
    for config in configs:
        for dataset in datasets:
            for label in labels:
                result = _put_one_exp(
                    Experiment(
                        config=config,
                        dataset=dataset,
                        label=label,
                        train_events=[],
                        valid_events=[],
                        test_events=test_events
                    )
                )
                experiments.append(exp_detail(eid=result, sha1=hash_config(json.loads(config)), dataset=dataset,
                                              label=label))

    # find by a property and group by different reduction dims
    for prop, values in {'dataset': datasets, 'label': labels}.items():
        for value in values:
            for reduction_dim in ['sha1', 'eid', 'dataset', 'label']:
                assert _test_reduction_dim(prop=prop, value=value, reduction_dim=reduction_dim, experiments=experiments)

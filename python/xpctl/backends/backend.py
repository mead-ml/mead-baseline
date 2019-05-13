from copy import deepcopy
import shutil
from collections import namedtuple
import json
import os
import numpy as np
from flask import abort

from baseline.utils import export, write_config_file

from xpctl.server.models import Experiment as ServerExperiment
from xpctl.server.models import Result as ServerResult
from xpctl.server.models import ExperimentAggregate as ServerExperimentAggregate
from xpctl.server.models import Response as ServerResponse
from xpctl.server.models import TaskSummary as ServerTaskSummary
from xpctl.server.models import AggregateResult as ServerAggregateResult
from xpctl.server.models import AggregateResultValues


__all__ = []
exporter = export(__all__)

TRAIN_EVENT = 'train_events'
VALID_EVENT = 'valid_events'
TEST_EVENT = 'test_events'
EVENT_TYPES = [TRAIN_EVENT, VALID_EVENT, TEST_EVENT]


METRICS_SORT_ASCENDING = ['avg_loss', 'perplexity']


@exporter
class Result(object):
    def __init__(self, metric, value, tick_type, tick, phase):
        super(Result, self).__init__()
        self.metric = metric
        self.value = value
        self.tick_type = tick_type
        self.tick = tick
        self.phase = phase
    
    def get_prop(self, field):
        if field not in self.__dict__:
            raise ValueError('{} does not have a property {}'.format(self.__class__, field))
        return self.__dict__[field]
        

@exporter
class AggregateResult(object):
    def __init__(self, metric, values):
        super(AggregateResult, self).__init__()
        self.metric = metric
        self.values = values

    def get_prop(self, field):
        if field not in self.__dict__:
            raise ValueError('{} does not have a property {}'.format(self.__class__, field))
        return self.__dict__[field]


@exporter
class Experiment(object):
    """ an experiment"""
    def __init__(self, train_events, valid_events, test_events, task, eid, username, hostname, config, exp_date, label,
                 dataset, sha1, version):
        super(Experiment, self).__init__()
        self.task = task
        self.train_events = train_events if train_events is not None else []
        self.valid_events = valid_events if valid_events is not None else []
        self.test_events = test_events if test_events is not None else []
        self.eid = eid
        self.username = username
        self.hostname = hostname
        self.config = config
        self.exp_date = exp_date
        self.label = label
        self.dataset = dataset
        self.exp_date = exp_date
        self.sha1 = sha1
        self.version = version
    
    def get_prop(self, field):
        if field not in self.__dict__:
            raise ValueError('{} does not have a property {}'.format(self.__class__, field))
        return self.__dict__[field]
    
    def add_result(self, result, event_type):
        if event_type == TRAIN_EVENT:
            self.train_events.append(result)
        elif event_type == VALID_EVENT:
            self.valid_events.append(result)
        elif event_type == TEST_EVENT:
            self.test_events.append(result)
        else:
            raise NotImplementedError('no handler for event type: [{}]'.format(event_type))


@exporter
class ExperimentSet(object):
    """ a list of experiment objects"""
    def __init__(self, data):
        super(ExperimentSet, self).__init__()
        self.data = data if data else []  # this should ideally be a set but the items are not hashable
        self.length = len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    def __iter__(self):
        for i in range(self.length):
            yield self.data[i]
    
    def __len__(self):
        return self.length

    def add_data(self, datum):
        """
        add a experiment data point
        :param datum:
        :return:
        """
        self.data.append(datum)
        self.length += 1
        
    def groupby(self, key):
        """ group the data points by key"""
        data_groups = {}
        if len(self.data) == 0:
            raise RuntimeError('Trying to group empty experiment set')
        task = self.data[0].get_prop('task')
        for datum in self.data:
            if datum.get_prop('task') != task:
                raise RuntimeError('Should not be grouping two experiments from different tasks')
            field = datum.get_prop(key)
            if field not in data_groups:
                data_groups[field] = ExperimentSet([datum])
            else:
                data_groups[field].add_data(datum)
        return ExperimentGroup(data_groups, key, task)
    
    def sort(self, key, reverse=True):
        """
        you can only sort when event_type is test, because there is only one data point
        :param key: metric to sort on
        :param reverse: reverse=True always except when key is avg_loss
        :return:
        """
        if key is None or key == 'None':
            return self
        test_results = [(index, [y for y in x.get_prop(TEST_EVENT) if y.metric == key][0]) for index, x in
                        enumerate(self.data)]
        test_results.sort(key=lambda x: x[1].value, reverse=reverse)
        final_results = []
        for index, _ in test_results:
            final_results.append(self.data[index])
        return ExperimentSet(data=final_results)


@exporter
class ExperimentGroup(object):
    """ a group of resultset objects"""
    def __init__(self, grouped_experiments, reduction_dim, task):
        super(ExperimentGroup, self).__init__()
        self.grouped_experiments = grouped_experiments
        self.reduction_dim = reduction_dim
        self.task = task
    
    def items(self):
        return self.grouped_experiments.items()
    
    def keys(self):
        return self.grouped_experiments.keys()
    
    def values(self):
        return self.grouped_experiments.values()
    
    def __iter__(self):
        for k, v in self.grouped_experiments.items():
            yield (k, v)
    
    def get(self, key):
        return self.grouped_experiments.get(key)
    
    def __len__(self):
        return len(self.grouped_experiments.keys())
    
    def reduce(self, aggregate_fns, event_type=TEST_EVENT):
        """ aggregate results across a result group"""
        data = {}
        num_experiments = {}
        for reduction_dim_value, experiments in self.grouped_experiments.items():
            num_experiments[reduction_dim_value] = len(experiments)
            data[reduction_dim_value] = {}
            for experiment in experiments:
                results = experiment.get_prop(event_type)
                for result in results:
                    if result.metric not in data[reduction_dim_value]:
                        data[reduction_dim_value][result.metric] = [result.value]
                    else:
                        data[reduction_dim_value][result.metric].append(result.value)
        # for each reduction dim value, (say when sha1 = x), all data[x][metric] lists should have the same length.
        for reduction_dim_value in data:
            lengths = []
            for metric in data[reduction_dim_value]:
                lengths.append(len(data[reduction_dim_value][metric]))
            try:
                assert len(set(lengths)) == 1
            except AssertionError:
                raise AssertionError('when reducing experiments over {}, for {}={}, the number of results are not the '
                                     'same over all metrics'.format(self.reduction_dim, self.reduction_dim,
                                                                    reduction_dim_value))
            
        aggregate_resultset = ExperimentAggregateSet(data=[])
        for reduction_dim_value in data:
            values = {}
            d = {self.reduction_dim: reduction_dim_value, 'num_exps': num_experiments[reduction_dim_value]}
            agr = deepcopy(ExperimentAggregate(task=self.task, **d))
            for metric in data[reduction_dim_value]:
                for fn_name, fn in aggregate_fns.items():
                    agg_value = fn(data[reduction_dim_value][metric])
                    values[fn_name] = agg_value
                agr.add_result(deepcopy(AggregateResult(metric=metric, values=values)), event_type=event_type)
            aggregate_resultset.add_data(agr)
        return aggregate_resultset
    

@exporter
class ExperimentAggregate(object):
    """ a result data point"""
    def __init__(self, task, train_events=[], valid_events=[], test_events=[], **kwargs):
        super(ExperimentAggregate, self).__init__()
        self.train_events = train_events if train_events is not None else []
        self.valid_events = valid_events if valid_events is not None else []
        self.test_events = test_events if test_events is not None else []
        self.task = task
        self.num_exps = kwargs.get('num_exps')
        self.eid = kwargs.get('eid')
        self.username = kwargs.get('username')
        self.label = kwargs.get('label')
        self.dataset = kwargs.get('dataset')
        self.exp_date = kwargs.get('exp_date')
        self.sha1 = kwargs.get('sha1')
    
    def get_prop(self, field):
        return self.__dict__[field]

    def add_result(self, aggregate_result, event_type):
        if event_type == TRAIN_EVENT:
            self.train_events.append(aggregate_result)
        elif event_type == VALID_EVENT:
            self.valid_events.append(aggregate_result)
        elif event_type == TEST_EVENT:
            self.test_events.append(aggregate_result)
        else:
            raise NotImplementedError('no handler for event type: [{}]'.format(event_type))
    

@exporter
class ExperimentAggregateSet(object):
    """ a list of aggregate result objects"""
    def __init__(self, data):
        super(ExperimentAggregateSet, self).__init__()
        self.data = data if data else []
        self.length = len(self.data)
    
    def add_data(self, data_point):
        """
        add a aggregateresult data point
        :param data_point:
        :return:
        """
        self.data.append(data_point)
        self.length += 1
    
    # TODO: add property annotations
    def __getitem__(self, i):
        return self.data[i]
    
    def __iter__(self):
        for i in range(self.length):
            yield self.data[i]
    
    def __len__(self):
        return self.length

    def sort(self, metric, aggregate_fn='avg', reverse=True):
        """
        you can only sort when event_type is test, because there is only one data point
        :param metric: metric to sort on
        :param aggregate_fn: this is an aggregate result, you have values for for different aggregate_fns. choose one
        :param reverse: reverse=True always except when key is avg_loss, perplexity
        :return:
        """
        if metric is None:
            return self
        test_results = [(index, [y for y in x.get_prop(TEST_EVENT) if y.metric == metric][0]) for index, x in
                        enumerate(self.data)]
        test_results.sort(key=lambda x: x[1].values[aggregate_fn], reverse=reverse)
        final_results = []
        for index, _ in test_results:
            final_results.append(self.data[index])
        return ExperimentSet(data=final_results)


@exporter
class TaskDatasetSummary(object):
    """ How many users experimented with this dataset in the given task?"""
    def __init__(self, task, dataset, experiment_set, user_num_exps=None):
        super(TaskDatasetSummary, self).__init__()
        self.task = task
        self.dataset = dataset
        if user_num_exps is not None:
            self.user_num_exps = user_num_exps
        else:
            exp_groups = experiment_set.groupby('username')
            self.user_num_exps = {username: len(exp_group)for username, exp_group in exp_groups}


@exporter
class TaskDatasetSummarySet(object):
    """ a list of TaskDatasetSummary objects."""
    def __init__(self, task, data):
        self.task = task
        self.data = data
    
    def groupby(self):
        """ group the TaskDatasetSummary objects. """
        d = {}
        for tdsummary in self.data:
            dataset = tdsummary.dataset
            d[dataset] = []
            for username in tdsummary.user_num_exps:
                d[dataset].append((username, tdsummary.user_num_exps[username]))
                
        return TaskSummary(self.task, d)
 

@exporter
class TaskSummary(object):
    def __init__(self, task, summary):
        self.task = task
        self.summary = summary


@exporter
class BackendResponse(object):
    def __init__(self, message, response_type, code=550):
        super(BackendResponse, self).__init__()
        self.message = message
        self.response_type = response_type
        self.code = code


@exporter
class BackendError(BackendResponse):
    def __init__(self, message, response_type="error", code=550):
        super(BackendError, self).__init__(message, response_type, code)
        self.message = message
        self.response_type = response_type
        self.code = code


@exporter
class BackendSuccess(BackendResponse):
    def __init__(self, message, response_type="success", code=250):
        super(BackendSuccess, self).__init__(message, response_type, code)
        self.message = message
        self.response_type = response_type
        self.code = code


@exporter
def log2json(log_file):
    s = []
    with open(log_file) as f:
        for line in f:
            x = line.replace("'", '"')
            s.append(json.loads(x))
    return s


@exporter
def json2log(events, log_file):
    with open(log_file, 'w') as wf:
        for event in events:
            wf.write(json.dumps(event)+'\n')


@exporter
def get_experiment_label(config_obj, task, **kwargs):
    if kwargs.get('label', None) is not None:
        return kwargs['label']
    if 'description' in config_obj:
        return config_obj['description']
    else:
        model_type = config_obj.get('model_type', 'default')
        backend = config_obj.get('backend', 'tensorflow')
        return "{}-{}-{}".format(task, backend, model_type)


@exporter
def safe_get(_object, key, alt):
    val = _object.get(key)
    if val is None or str(val) is None:
        return alt
    return val


@exporter
def is_error(_object):
    """
    poor man's type checking, apparently you can not do `type(_object) == BackendError` when you move that class in the
    same file
    :param _object:
    :return:
    """
    if hasattr(_object, 'response_type'):
        return True
    return False


@exporter
def serialize_experiment(exp):
    """
    serialize an Experiment object for consumption by swagger client
    :param exp: backends.backend.Experiment
    :return:
    """
    if is_error(exp):
        return abort(500, exp.message)
    train_events = [ServerResult(**r.__dict__) for r in exp.train_events]
    valid_events = [ServerResult(**r.__dict__) for r in exp.valid_events]
    test_events = [ServerResult(**r.__dict__) for r in exp.test_events]
    d = exp.__dict__
    d.update({'train_events': train_events})
    d.update({'valid_events': valid_events})
    d.update({'test_events': test_events})
    return ServerExperiment(**d)


@exporter
def serialize_experiment_aggregate(agg_exps):
    """
    serialize an ExperimentAggregate object for consumption by swagger client
    :param agg_exps: backends.backend.ExperimentAggregate
    :return:
    """
    if is_error(agg_exps):
        return abort(500, agg_exps.message)
    results = []
    for agg_exp in agg_exps:
        train_events = [ServerAggregateResult(metric=r.metric,
                                              values=[AggregateResultValues(k, v) for k, v in r.values.items()])
                        for r in agg_exp.train_events]
        valid_events = [ServerAggregateResult(metric=r.metric,
                                              values=[AggregateResultValues(k, v) for k, v in r.values.items()])
                        for r in agg_exp.valid_events]
        test_events = [ServerAggregateResult(metric=r.metric,
                                             values=[AggregateResultValues(k, v) for k, v in r.values.items()])
                       for r in agg_exp.test_events]
        
        d = agg_exp.__dict__
        d.update({'train_events': train_events})
        d.update({'valid_events': valid_events})
        d.update({'test_events': test_events})
        results.append(ServerExperimentAggregate(**d))
    return results


@exporter
def serialize_experiment_list(exps):
    """
    serialize a list of Experiment objects for consumption by swagger client
    :param exps:
    :return:
    """
    if is_error(exps):
        return abort(500, exps.message)
    results = []
    for exp in exps:
        if type(exp) == BackendError:
            return BackendResponse(**exp.__dict__)
        train_events = [ServerResult(**r.__dict__) for r in exp.train_events]
        valid_events = [ServerResult(**r.__dict__) for r in exp.valid_events]
        test_events = [ServerResult(**r.__dict__) for r in exp.test_events]
        d = exp.__dict__
        d.update({'train_events': train_events})
        d.update({'valid_events': valid_events})
        d.update({'test_events': test_events})
        results.append(ServerExperiment(**d))
    return results


@exporter
def serialize_task_summary(task_summary):
    """
    serialize a TaskSummary object for consumption by swagger client
    :param task_summary:
    :return:
    """
    if is_error(task_summary):
        return abort(500, task_summary.message)
    return ServerTaskSummary(**task_summary.__dict__)


@exporter
def serialize_task_summary_list(task_summaries):
    _task_summaries = []
    for task_summary in task_summaries:
        if not is_error(task_summary):  # should we abort if we cant get summary for a task in the database?
            _task_summaries.append(ServerTaskSummary(**task_summary.__dict__))
    return _task_summaries


@exporter
def serialize_dict(config):
    """
    Serializes a dict, exists only to handle the error case
    :param config:
    :return:
    """
    if is_error(config):
        return abort(500, config.message)
    return config


@exporter
def serialize_response(result):
    """
    serializes a Response object for swagger client consumption
    :param result:
    :return:
    """
    return ServerResponse(**result.__dict__)


@exporter
def deserialize_result(result):
    return Result(
        metric=result.metric,
        value=result.value,
        tick_type=result.tick_type,
        tick=result.tick,
        phase=result.phase
    )


@exporter
def pack_results_in_events(results):
    d = {}
    for result in results:
        if result.tick not in d:
            d[result.tick] = {result.metric: result.value,
                              'tick_type': result.tick_type,
                              'phase': result.phase,
                              'tick': result.tick
                              }
        else:
            d[result.tick].update({result.metric: result.value})
    return list(d.values())


@exporter
def client_experiment_to_put_result_consumable(exp):
    train_events = pack_results_in_events(exp.train_events)
    valid_events = pack_results_in_events(exp.valid_events)
    test_events = pack_results_in_events(exp.test_events)
    config = exp.config
    task = exp.task
    extra_args = {
        'sha1': exp.sha1,
        'dataset': exp.dataset,
        'username': exp.username,
        'hostname': exp.hostname,
        'exp_date': exp.exp_date,
        'label': exp.label
    }
    put_result_consumable = namedtuple('put_result_consumable', ['task', 'config_obj', 'events_obj', 'extra_args'])
    return put_result_consumable(task=task, config_obj=json.loads(config),
                                 events_obj=train_events+valid_events+test_events,
                                 extra_args=extra_args)


@exporter
def aggregate_results(resultset, reduction_dim, event_type, num_exps_per_reduction):
    # TODO: implement a trim method for ExperimentGroup
    grouped_result = resultset.groupby(reduction_dim)
    aggregate_fns = {'min': np.min, 'max': np.max, 'avg': np.mean, 'std': np.std}
    return grouped_result.reduce(aggregate_fns=aggregate_fns, event_type=event_type)


@exporter
def write_experiment(experiment, basedir):
    eid = str(experiment.eid)
    basedir = os.path.join(basedir, eid)
    os.makedirs(basedir)
    train_events = pack_results_in_events(experiment.train_events)
    valid_events = pack_results_in_events(experiment.valid_events)
    test_events = pack_results_in_events(experiment.test_events)
    json2log(train_events + valid_events + test_events, os.path.join(basedir, '{}-reporting.log'.format(eid)))
    config = json.loads(experiment.config) if type(experiment.config) is str else experiment.config
    write_config_file(config, os.path.join(basedir, '{}-config.yml'.format(eid)))
    d = experiment.__dict__
    [d.pop(event_type) for event_type in EVENT_TYPES]
    d.pop('config')
    write_config_file(d, os.path.join(basedir, '{}-meta.yml'.format(eid)))

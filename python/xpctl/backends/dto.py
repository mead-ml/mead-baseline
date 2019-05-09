from collections import namedtuple
import json
import os
import numpy as np
from flask import abort

from baseline.utils import export, write_config_file

from xpctl.swagger_server.models import Experiment as ServerExperiment
from xpctl.swagger_server.models import Result as ServerResult
from xpctl.swagger_server.models import ExperimentAggregate, Response, AggregateResult, TaskSummary, \
    AggregateResultValues
from xpctl.backends.data import Experiment, Error, Result, Success
from xpctl.backends.helpers import json2log
from xpctl.backends.data import EVENT_TYPES


__all__ = []
exporter = export(__all__)


@exporter
def serialize_experiment_details(exp):
    if type(exp) == Error:
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
def serialize_get_results(agg_exps):
    if type(agg_exps) == Error:
        return abort(500, agg_exps.message)
    results = []
    for agg_exp in agg_exps:
        train_events = [AggregateResult(metric=r.metric,
                                        values=[AggregateResultValues(k, v) for k, v in r.values.items()])
                        for r in agg_exp.train_events]
        valid_events = [AggregateResult(metric=r.metric,
                                        values=[AggregateResultValues(k, v) for k, v in r.values.items()])
                        for r in agg_exp.valid_events]
        test_events = [AggregateResult(metric=r.metric,
                                       values=[AggregateResultValues(k, v) for k, v in r.values.items()])
                       for r in agg_exp.test_events]

        d = agg_exp.__dict__
        d.update({'train_events': train_events})
        d.update({'valid_events': valid_events})
        d.update({'test_events': test_events})
        results.append(ExperimentAggregate(**d))
    return results


@exporter
def serialize_list_results(exps):
    if type(exps) == Error:
        return abort(500, exps.message)
    results = []
    for exp in exps:
        if type(exp) == Error:
            return Response(**exp.__dict__)
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
    if type(task_summary) == Error:
        return abort(500, task_summary.message)
    return TaskSummary(**task_summary.__dict__)


@exporter
def serialize_summary(task_summaries):
    _task_summaries = []
    for task_summary in task_summaries:
        if type(task_summary) != Error:  # should we abort if we cant get summary for a task in the database?
            _task_summaries.append(TaskSummary(**task_summary.__dict__))
    return _task_summaries


@exporter
def serialize_config2json(config):
    if type(config) == Error:
        return abort(500, config.message)
    return config


@exporter
def serialize_get_model_location(location):
    return Response(**location.__dict__)


@exporter
def serialize_post_requests(result):
    return Response(**result.__dict__)


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
def deserialize_experiment(exp):
    train_events = [deserialize_result(r) for r in exp.train_events]
    valid_events = [deserialize_result(r) for r in exp.valid_events]
    test_events = [deserialize_result(r) for r in exp.test_events]
    return Experiment(
        task=exp.task,
        eid=exp.eid,
        username=exp.username,
        hostname=exp.hostname,
        config=exp.config,
        exp_date=exp.exp_date,
        label=exp.label,
        dataset=exp.dataset,
        sha1=exp.sha1,
        version=exp.version,
        train_events=train_events,
        valid_events=valid_events,
        test_events=test_events
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
def experiment_to_put_result_consumable(exp):
    d = exp.__dict__
    train_events = pack_results_in_events(exp.train_events)
    d.pop('train_events')
    valid_events = pack_results_in_events(exp.valid_events)
    d.pop('valid_events')
    test_events = pack_results_in_events(exp.test_events)
    d.pop('test_events')
    config = exp.config
    d.pop('config')
    task = exp.task
    d.pop('task')
    put_result_consumable = namedtuple('put_result_consumable', ['task', 'config_obj', 'events_obj', 'extra_args'])
    return put_result_consumable(task=task, config_obj=json.loads(config),
                                 events_obj=train_events+valid_events+test_events,
                                 extra_args=d)


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

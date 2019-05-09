from swagger_server.models import Experiment, ExperimentAggregate, Result, Response, AggregateResult, TaskSummary, \
    AggregateResultValues
from xpctl.backend.data import Error
from xpctl import backend
from flask import abort
from collections import namedtuple
import json


def dto_experiment_details(exp):
    if type(exp) == Error:
        return abort(500, exp.message)
    train_events = [Result(**r.__dict__) for r in exp.train_events]
    valid_events = [Result(**r.__dict__) for r in exp.valid_events]
    test_events = [Result(**r.__dict__) for r in exp.test_events]
    d = exp.__dict__
    d.update({'train_events': train_events})
    d.update({'valid_events': valid_events})
    d.update({'test_events': test_events})
    return Experiment(**d)


def dto_get_results(agg_exps):
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


def dto_list_results(exps):
    if type(exps) == Error:
        return abort(500, exps.message)
    results = []
    for exp in exps:
        if type(exp) == Error:
            return Response(**exp.__dict__)
        train_events = [Result(**r.__dict__) for r in exp.train_events]
        valid_events = [Result(**r.__dict__) for r in exp.valid_events]
        test_events = [Result(**r.__dict__) for r in exp.test_events]
        d = exp.__dict__
        d.update({'train_events': train_events})
        d.update({'valid_events': valid_events})
        d.update({'test_events': test_events})
        results.append(Experiment(**d))
    return results


def dto_task_summary(task_summary):
    if type(task_summary) == Error:
        return abort(500, task_summary.message)
    return TaskSummary(**task_summary.__dict__)


def dto_summary(task_summaries):
    _task_summaries = []
    for task_summary in task_summaries:
        if type(task_summary) != Error:  # should we abort if we cant get summary for a task in the database?
            _task_summaries.append(TaskSummary(**task_summary.__dict__))
    return _task_summaries


def dto_config2json(config):
    if type(config) == Error:
        return abort(500, config.message)
    return config


def dto_get_model_location(location):
    return Response(**location.__dict__)


def dto_put_requests(result):
    return Response(**result.__dict__)


def convert_to_data_result(result: Result):
    return backend.data.Result(
        metric=result.metric,
        value=result.value,
        tick_type=result.tick_type,
        tick=result.tick,
        phase=result.phase
    )
    
    
def dto_to_experiment(exp):
    train_events = [convert_to_data_result(r) for r in exp.train_events]
    valid_events = [convert_to_data_result(r) for r in exp.valid_events]
    test_events = [convert_to_data_result(r) for r in exp.test_events]
    return backend.data.Experiment(
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


def pack_events(results):
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


def unpack_experiment(exp):
    d = exp.__dict__
    train_events = pack_events(exp.train_events)
    d.pop('train_events')
    valid_events = pack_events(exp.valid_events)
    d.pop('valid_events')
    test_events = pack_events(exp.test_events)
    d.pop('test_events')
    config = exp.config
    d.pop('config')
    task = exp.task
    d.pop('task')
    unpacked_mongo_result = namedtuple('unpacked_mongo_result', ['task', 'config_obj', 'events_obj', 'extra_args'])
    return unpacked_mongo_result(task=task, config_obj=json.loads(config), events_obj=train_events+valid_events+test_events,
                                 extra_args=d)

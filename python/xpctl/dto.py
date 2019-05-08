from swagger_server.models import Experiment, ExperimentAggregate, Result, Response, AggregateResult, TaskSummary, \
    AggregateResultValues
import xpctl.data
from flask import abort


def dto_experiment_details(exp):
    if type(exp) == xpctl.data.Error:
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
    if type(agg_exps) == xpctl.data.Error:
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
    if type(exps) == xpctl.data.Error:
        return abort(500, exps.message)
    results = []
    for exp in exps:
        if type(exp) == xpctl.data.Error:
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
    if type(task_summary) == xpctl.data.Error:
        return abort(500, task_summary.message)
    return TaskSummary(**task_summary.__dict__)


def dto_summary(task_summaries):
    if type(task_summaries) == xpctl.data.Error:
        return abort(500, task_summaries.message)
    return [TaskSummary(**task_summary.__dict__) for task_summary in task_summaries]


def dto_config2json(config):
    if type(config) == xpctl.data.Error:
        return Response(config.__dict__)
    return config


def dto_get_model_location(location):
    return Response(**location.__dict__)


def dto_put_requests(result):
    return Response(**result.__dict__)


def dto_to_experiment(exp):
    train_events = [xpctl.data.Result(**r.__dict__) for r in exp.train_events]
    valid_events = [xpctl.data.Result(**r.__dict__) for r in exp.valid_events]
    test_events = [xpctl.data.Result(**r.__dict__) for r in exp.test_events]
    d = exp.__dict__
    d.update({'train_events': train_events})
    d.update({'valid_events': valid_events})
    d.update({'test_events': test_events})
    return xpctl.data.Experiment(**d)

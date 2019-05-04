from swagger_server.models import Experiment, ExperimentAggregate, Result, Error, AggregateResult
from xpctl.backend.mongo.dto import MongoError


def dto_experiment_details(exp, task):
    if type(exp) == MongoError:
        return Error(code=exp.code, message=exp.message)
    train_events = [Result(**r.__dict__) for r in exp.train_events]
    dev_events = [Result(**r.__dict__) for r in exp.dev_events]
    test_events = [Result(**r.__dict__) for r in exp.test_events]
    d = exp.__dict__
    d.update({'task': task})
    d.update({'train_events': train_events})
    d.update({'dev_events': dev_events})
    d.update({'test_events': test_events})
    return Experiment(**d)


def dto_get_results(agg_exps, task):
    if type(agg_exps) == MongoError:
        return Error(code=agg_exps.code, message=agg_exps.message)
    results = []
    for agg_exp in agg_exps:
        if type(agg_exp) == MongoError:
            return Error(code=agg_exp.code, message=agg_exp.message)
        train_events = [AggregateResult(**r.__dict__) for r in agg_exp.train_events]
        dev_events = [AggregateResult(**r.__dict__) for r in agg_exp.dev_events]
        test_events = [AggregateResult(**r.__dict__) for r in agg_exp.test_events]
        d = agg_exp.__dict__
        d.update({'task': task})
        d.update({'train_events': train_events})
        d.update({'dev_events': dev_events})
        d.update({'test_events': test_events})
        results.append(ExperimentAggregate(**d))
    return results

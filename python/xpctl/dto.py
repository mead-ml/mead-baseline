from swagger_server.models import Experiment, ExperimentAggregate, Result, Error, AggregateResult
from xpctl.backend.mongo.dto import MongoError

def dto_experiment_details(exp, task):
    if type(exp) == MongoError:
        return Error(code=exp.code, message=exp.message)
    train_results = [Result(**r.__dict__) for r in exp.train_results]
    dev_results = [Result(**r.__dict__) for r in exp.dev_results]
    test_results = [Result(**r.__dict__) for r in exp.test_results]
    d = exp.__dict__
    d.update({'task': task})
    d.update({'train_results': train_results})
    d.update({'dev_results': dev_results})
    d.update({'test_results': test_results})
    return Experiment(**d)


def dto_get_results(agg_exps, task):
    results = []
    for agg_exp in agg_exps:
        if type(agg_exp) == MongoError:
            return Error(code=agg_exp.code, message=agg_exp.message)
        train_results = [AggregateResult(**r.__dict__) for r in agg_exp.train_results]
        dev_results = [AggregateResult(**r.__dict__) for r in agg_exp.dev_results]
        test_results = [AggregateResult(**r.__dict__) for r in agg_exp.test_results]
        d = agg_exp.__dict__
        d.update({'task': task})
        d.update({'train_results': train_results})
        d.update({'dev_results': dev_results})
        d.update({'test_results': test_results})
        results.append(ExperimentAggregate(**d))
    return results

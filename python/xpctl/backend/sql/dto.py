from xpctl.backend.data import Experiment, Result, ExperimentSet, Error, aggregate_results


def event2phase(event_type):
    if event_type == 'train_events':
        return 'Train'
    if event_type == 'valid_events':
        return 'Valid'
    if event_type == 'test_events':
        return 'Test'
    Error(message='Unknown event type {}'.format(event_type))


def get_filtered_metrics(metrics_from_db, metrics_from_user):
    if not metrics_from_user:
        metrics = list(metrics_from_db)
    elif metrics_from_user - metrics_from_db:
        return Error(message='Metrics [{}] not found'.format(','.join(list(metrics_from_user - metrics_from_db))))
    else:
        metrics = list(metrics_from_user)
    return metrics


def get_sql_metrics(event):
    return set([r.metric for r in event.results])
    
    
def flatten(_list):
    return [item for sublist in _list for item in sublist]


def create_results(event, metrics):
    results = []
    for r in event.results:
        if r.metric in metrics:
            results.append(
                Result(metric=r.metric, value=r.value, tick_type=event.tick_type, tick=event.tick, phase=event.phase)
            )
    return results

    
def sql_result_to_data_experiment(exp, event_type, metrics):
    phase = event2phase(event_type)
    if type(phase) is Error:
        return phase
    phase_events = [event for event in exp.events if event.phase == phase]
    if len(phase_events) == 0:
        return Error('experiment id {} has 0 {}'.format(exp.eid, event_type))
    metrics = get_filtered_metrics(get_sql_metrics(phase_events[0]), set(metrics))
    if type(metrics) is Error:
        return metrics
    results = flatten([create_results(event, metrics) for event in phase_events])
    _exp = Experiment(
        task=exp.task,
        eid=exp.eid,
        username=exp.username,
        hostname=exp.hostname,
        config=exp.config,
        exp_date=exp.date,
        label=exp.label,
        dataset=exp.dataset,
        sha1=exp.sha1,
        version=exp.version,
        train_events=[],
        valid_events=[],
        test_events=[]
    )
    for r in results:
        _exp.add_result(r, event_type)
    return _exp


def get_data_experiment_set(data_experiments):
    return ExperimentSet(data_experiments)


def aggregate_sql_results(data_experiments, reduction_dim, event_type, numexp_reduction_dim):
    experiment_set = get_data_experiment_set(data_experiments)
    return aggregate_results(experiment_set, reduction_dim, event_type, numexp_reduction_dim)


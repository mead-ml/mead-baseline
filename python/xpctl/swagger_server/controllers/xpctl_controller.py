import connexion

from swagger_server.models.experiment import Experiment  # noqa: E501
import flask
from backend.dto import *


def config2json(task, sha1):  # noqa: E501
    """get config for sha1

    config for sha1 # noqa: E501

    :param task: task
    :type task: str
    :param sha1: sha1
    :type sha1: str

    :rtype: object
    """
    backend = flask.globals.current_app.backend
    return dto_config2json(backend.config2json(task, sha1))


def experiment_details(task, eid, event_type=None, metric=None):  # noqa: E501
    """Find experiment by id

    Returns a single experiment # noqa: E501

    :param task: task name
    :type task: str
    :param eid: ID of experiment to return
    :type eid: str
    :param event_type: 
    :type event_type: str
    :param metric: 
    :type metric: List[str]

    :rtype: Experiment
    """
    backend = flask.globals.current_app.backend
    return dto_experiment_details(backend.get_experiment_details(task, eid, event_type, metric))


def get_model_location(task, eid):  # noqa: E501
    """get model loc for experiment

    get model loc for experiment # noqa: E501

    :param task: task
    :type task: str
    :param eid: experiment id
    :type eid: str

    :rtype: Response
    """
    backend = flask.globals.current_app.backend
    return dto_get_model_location(backend.get_model_location(task, eid))


def get_results_by_dataset(task, dataset, reduction_dim=None, metric=None, sort=None, numexp_reduction_dim=None, event_type=None):  # noqa: E501
    """Find results by dataset and task

    Returns a single experiment # noqa: E501

    :param task: task name
    :type task: str
    :param dataset: dataset name
    :type dataset: str
    :param reduction_dim: which dimension to reduce on, default&#x3D;sha1
    :type reduction_dim: str
    :param metric: metric
    :type metric: List[str]
    :param sort: metric to sort results on
    :type sort: str
    :param numexp_reduction_dim: max number of experiments in an aggregate group
    :type numexp_reduction_dim: int
    :param event_type: train/dev/test
    :type event_type: str

    :rtype: List[ExperimentAggregate]
    """
    backend = flask.globals.current_app.backend
    return dto_get_results(backend.get_results(task, 'dataset', dataset, reduction_dim, metric, sort,
                                               numexp_reduction_dim, event_type))


def get_results_by_prop(task, prop, value, reduction_dim=None, metric=None, sort=None, numexp_reduction_dim=None, event_type=None):  # noqa: E501
    """Find results by property and value

    Find results by property and value # noqa: E501

    :param task: task name
    :type task: str
    :param prop: property of an experiment dataset, username, label etc
    :type prop: str
    :param value: value of the property. eg: prop&#x3D;username&amp;value&#x3D;dpressel
    :type value: str
    :param reduction_dim: which dimension to reduce on, default&#x3D;sha1
    :type reduction_dim: str
    :param metric: metric
    :type metric: List[str]
    :param sort: metric to sort results on
    :type sort: str
    :param numexp_reduction_dim: max number of experiments in an aggregate group
    :type numexp_reduction_dim: int
    :param event_type: train/dev/test
    :type event_type: str

    :rtype: List[ExperimentAggregate]
    """
    backend = flask.globals.current_app.backend
    return dto_get_results(backend.get_results(task, prop, value, reduction_dim, metric, sort,
                                               numexp_reduction_dim, event_type))


def list_experiments_by_prop(task, prop, value, user=None, metric=None, sort=None, event_type=None):  # noqa: E501
    """list all experiments for this property and value

    list all experiments for this property (sha1/ username) and value (1cd21df91770b4dbed64a683558b062e3dee61f0/ dpressel) # noqa: E501

    :param task: task name
    :type task: str
    :param prop: property: username, dataset
    :type prop: str
    :param value: value: dpressel, SST2
    :type value: str
    :param user: 
    :type user: List[str]
    :param metric: 
    :type metric: List[str]
    :param sort: 
    :type sort: str
    :param event_type: 
    :type event_type: str

    :rtype: List[Experiment]
    """
    backend = flask.globals.current_app.backend
    return dto_list_results(backend.list_results(task, prop, value, user, metric, sort, event_type))


def list_experiments_by_sha1(task, sha1, user=None, metric=None, sort=None, event_type=None):  # noqa: E501
    """list all experiments for this sha1

    list all experiments for this sha1 # noqa: E501

    :param task: task name
    :type task: str
    :param sha1: sha1
    :type sha1: str
    :param user: 
    :type user: List[str]
    :param metric: 
    :type metric: List[str]
    :param sort: 
    :type sort: str
    :param event_type: 
    :type event_type: str

    :rtype: List[Experiment]
    """
    backend = flask.globals.current_app.backend
    return dto_list_results(backend.list_results(task, 'sha1', sha1, user, metric, sort, event_type))


def put_result(task, experiment, user=None, label=None):  # noqa: E501
    """Add a new experiment in database

     # noqa: E501

    :param task: 
    :type task: str
    :param experiment: 
    :type experiment: dict | bytes
    :param user: 
    :type user: str
    :param label: 
    :type label: str

    :rtype: Response
    """
    if connexion.request.is_json:
        experiment = Experiment.from_dict(connexion.request.get_json())  # noqa: E501
    backend = flask.globals.current_app.backend
    return dto_put_requests(backend.put_result(task, dto_to_experiment(experiment)))


def remove_experiment(task, eid):  # noqa: E501
    """delete an experiment from the database

     # noqa: E501

    :param task: 
    :type task: str
    :param eid: 
    :type eid: str

    :rtype: Response
    """
    backend = flask.globals.current_app.backend
    return dto_put_requests(backend.remove_experiment(task, eid))


def summary():  # noqa: E501
    """get summary for task

    summary for task # noqa: E501


    :rtype: List[TaskSummary]
    """
    backend = flask.globals.current_app.backend
    return dto_summary(backend.summary())


def task_summary(task):  # noqa: E501
    """get summary for task

    summary for task # noqa: E501

    :param task: task
    :type task: str

    :rtype: TaskSummary
    """
    backend = flask.globals.current_app.backend
    return dto_task_summary(backend.task_summary(task))


def update_label(task, eid, label):  # noqa: E501
    """update label for an experiment

     # noqa: E501

    :param task: 
    :type task: str
    :param eid: 
    :type eid: str
    :param label: 
    :type label: str

    :rtype: Response
    """
    backend = flask.globals.current_app.backend
    return dto_put_requests(backend.update_prop(task, eid, prop='label', value=label))
   

def update_property(task, eid, prop, value):  # noqa: E501
    """update property for an experiment

     # noqa: E501

    :param task: 
    :type task: str
    :param eid: 
    :type eid: str
    :param prop: 
    :type prop: str
    :param value: 
    :type value: str

    :rtype: Response
    """
    backend = flask.globals.current_app.backend
    return dto_put_requests(backend.update_prop(task, eid, prop, value))


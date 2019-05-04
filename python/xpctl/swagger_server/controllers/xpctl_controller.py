import connexion
import six

from swagger_server.models.error import Error  # noqa: E501
from swagger_server.models.experiment import Experiment  # noqa: E501
from swagger_server.models.experiment_aggregate import ExperimentAggregate  # noqa: E501
from swagger_server import util
import flask
from xpctl.dto import *


def experiment_details(task, eid):  # noqa: E501
    """Find experiment by id

    Returns a single experiment # noqa: E501

    :param task: task name
    :type task: str
    :param eid: ID of experiment to return
    :type eid: str

    :rtype: Experiment
    """
    backend = flask.globals.current_app.backend
    return dto_experiment_details(backend.single_experiment(task, eid), task)


def find_by_label(label):  # noqa: E501
    """Finds experiment by label

    Multiple label values can be provided with comma separated strings # noqa: E501

    :param label: label values that need to be considered for filter
    :type label: List[str]

    :rtype: List[Experiment]
    """
    return 'do some magic!'


def get_results(task, prop, value, reduction_dim=None, metric=None, sort=None, nconfig=None, event_type=None):  # noqa: E501
    """Find results by dataset and task

    Returns a single experiment # noqa: E501

    :param task: task name
    :type task: str
    :param prop: property of an experiment dataset, username, label etc
    :type prop: str
    :param value: value of the property. eg: prop&#x3D;username&amp;value&#x3D;dpressel
    :type value: str
    :param reduction_dim: which dimension to reduce on, default&#x3D;sha1
    :type reduction_dim: str
    :param metric: metric
    :type metric: str
    :param sort: metric to sort results on
    :type sort: str
    :param nconfig: number of experiments to aggregate
    :type nconfig: int
    :param event_type: train/dev/test
    :type event_type: str

    :rtype: List[ExperimentAggregate]
    """
    backend = flask.globals.current_app.backend
    return dto_get_results(backend.get_results(task, prop, value, metric, sort, nconfig, event_type, reduction_dim),
                           task)


def put_result(experiment):  # noqa: E501
    """Add a new experiment in database

     # noqa: E501

    :param experiment: new mead experiment
    :type experiment: dict | bytes

    :rtype: None
    """
    if connexion.request.is_json:
        experiment = Experiment.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'


def remove_experiment(task, eid):  # noqa: E501
    """Deletes an experiment

     # noqa: E501

    :param task: task name
    :type task: str
    :param eid: experiment id to delete
    :type eid: str

    :rtype: None
    """
    return 'do some magic!'


def update_label(task, eid, label=None):  # noqa: E501
    """Updates an experiment in the database with form data

     # noqa: E501

    :param task: task name
    :type task: str
    :param eid: ID of experiment that needs to be updated
    :type eid: str
    :param label: Updated label of the experiment
    :type label: str

    :rtype: None
    """
    return 'do some magic!'

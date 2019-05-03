import connexion
import six

from swagger_server.models.error import Error  # noqa: E501
from swagger_server.models.experiment import Experiment  # noqa: E501
from swagger_server import util
import flask


def experiment_details(task, _id):  # noqa: E501
    """Find experiment by id

    Returns a single experiment # noqa: E501

    :param task: task name
    :type task: str
    :param _id: ID of experiment to return
    :type _id: str

    :rtype: Experiment
    """
    backend = flask.globals.current_app.backend
    return backend.single_experiment(task, _id)


def find_by_label(label):  # noqa: E501
    """Finds experiment by label

    Multiple label values can be provided with comma separated strings # noqa: E501

    :param label: label values that need to be considered for filter
    :type label: List[str]

    :rtype: List[Experiment]
    """
    return 'do some magic!'


def get_results(task, dataset):  # noqa: E501
    """Find results by dataset and task

    Returns a single experiment # noqa: E501

    :param task: task name
    :type task: str
    :param dataset: dataset name
    :type dataset: str

    :rtype: Experiment
    """
    return 'do some magic!'


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


def remove_experiment(task, _id):  # noqa: E501
    """Deletes an experiment

     # noqa: E501

    :param task: task name
    :type task: str
    :param _id: experiment id to delete
    :type _id: str

    :rtype: None
    """
    return 'do some magic!'


def update_label(task, _id, label=None):  # noqa: E501
    """Updates an experiment in the database with form data

     # noqa: E501

    :param task: task name
    :type task: str
    :param _id: ID of experiment that needs to be updated
    :type _id: str
    :param label: Updated label of the experiment
    :type label: str

    :rtype: None
    """
    return 'do some magic!'

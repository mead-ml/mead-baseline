# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server.models.result import Result  # noqa: F401,E501
from swagger_server import util


class Experiment(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(self, task=None, eid=None, sha1=None, config=None, dataset=None, username=None, hostname=None, exp_date=None, label=None, version=None, train_events=None, valid_events=None, test_events=None):  # noqa: E501
        """Experiment - a model defined in Swagger

        :param task: The task of this Experiment.  # noqa: E501
        :type task: str
        :param eid: The eid of this Experiment.  # noqa: E501
        :type eid: str
        :param sha1: The sha1 of this Experiment.  # noqa: E501
        :type sha1: str
        :param config: The config of this Experiment.  # noqa: E501
        :type config: str
        :param dataset: The dataset of this Experiment.  # noqa: E501
        :type dataset: str
        :param username: The username of this Experiment.  # noqa: E501
        :type username: str
        :param hostname: The hostname of this Experiment.  # noqa: E501
        :type hostname: str
        :param exp_date: The exp_date of this Experiment.  # noqa: E501
        :type exp_date: str
        :param label: The label of this Experiment.  # noqa: E501
        :type label: str
        :param version: The version of this Experiment.  # noqa: E501
        :type version: str
        :param train_events: The train_events of this Experiment.  # noqa: E501
        :type train_events: List[Result]
        :param valid_events: The valid_events of this Experiment.  # noqa: E501
        :type valid_events: List[Result]
        :param test_events: The test_events of this Experiment.  # noqa: E501
        :type test_events: List[Result]
        """
        self.swagger_types = {
            'task': str,
            'eid': str,
            'sha1': str,
            'config': str,
            'dataset': str,
            'username': str,
            'hostname': str,
            'exp_date': str,
            'label': str,
            'version': str,
            'train_events': List[Result],
            'valid_events': List[Result],
            'test_events': List[Result]
        }

        self.attribute_map = {
            'task': 'task',
            'eid': 'eid',
            'sha1': 'sha1',
            'config': 'config',
            'dataset': 'dataset',
            'username': 'username',
            'hostname': 'hostname',
            'exp_date': 'exp_date',
            'label': 'label',
            'version': 'version',
            'train_events': 'train_events',
            'valid_events': 'valid_events',
            'test_events': 'test_events'
        }

        self._task = task
        self._eid = eid
        self._sha1 = sha1
        self._config = config
        self._dataset = dataset
        self._username = username
        self._hostname = hostname
        self._exp_date = exp_date
        self._label = label
        self._version = version
        self._train_events = train_events
        self._valid_events = valid_events
        self._test_events = test_events

    @classmethod
    def from_dict(cls, dikt):
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The Experiment of this Experiment.  # noqa: E501
        :rtype: Experiment
        """
        return util.deserialize_model(dikt, cls)

    @property
    def task(self):
        """Gets the task of this Experiment.


        :return: The task of this Experiment.
        :rtype: str
        """
        return self._task

    @task.setter
    def task(self, task):
        """Sets the task of this Experiment.


        :param task: The task of this Experiment.
        :type task: str
        """

        self._task = task

    @property
    def eid(self):
        """Gets the eid of this Experiment.


        :return: The eid of this Experiment.
        :rtype: str
        """
        return self._eid

    @eid.setter
    def eid(self, eid):
        """Sets the eid of this Experiment.


        :param eid: The eid of this Experiment.
        :type eid: str
        """

        self._eid = eid

    @property
    def sha1(self):
        """Gets the sha1 of this Experiment.


        :return: The sha1 of this Experiment.
        :rtype: str
        """
        return self._sha1

    @sha1.setter
    def sha1(self, sha1):
        """Sets the sha1 of this Experiment.


        :param sha1: The sha1 of this Experiment.
        :type sha1: str
        """

        self._sha1 = sha1

    @property
    def config(self):
        """Gets the config of this Experiment.


        :return: The config of this Experiment.
        :rtype: str
        """
        return self._config

    @config.setter
    def config(self, config):
        """Sets the config of this Experiment.


        :param config: The config of this Experiment.
        :type config: str
        """

        self._config = config

    @property
    def dataset(self):
        """Gets the dataset of this Experiment.


        :return: The dataset of this Experiment.
        :rtype: str
        """
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        """Sets the dataset of this Experiment.


        :param dataset: The dataset of this Experiment.
        :type dataset: str
        """

        self._dataset = dataset

    @property
    def username(self):
        """Gets the username of this Experiment.


        :return: The username of this Experiment.
        :rtype: str
        """
        return self._username

    @username.setter
    def username(self, username):
        """Sets the username of this Experiment.


        :param username: The username of this Experiment.
        :type username: str
        """

        self._username = username

    @property
    def hostname(self):
        """Gets the hostname of this Experiment.


        :return: The hostname of this Experiment.
        :rtype: str
        """
        return self._hostname

    @hostname.setter
    def hostname(self, hostname):
        """Sets the hostname of this Experiment.


        :param hostname: The hostname of this Experiment.
        :type hostname: str
        """

        self._hostname = hostname

    @property
    def exp_date(self):
        """Gets the exp_date of this Experiment.


        :return: The exp_date of this Experiment.
        :rtype: str
        """
        return self._exp_date

    @exp_date.setter
    def exp_date(self, exp_date):
        """Sets the exp_date of this Experiment.


        :param exp_date: The exp_date of this Experiment.
        :type exp_date: str
        """

        self._exp_date = exp_date

    @property
    def label(self):
        """Gets the label of this Experiment.


        :return: The label of this Experiment.
        :rtype: str
        """
        return self._label

    @label.setter
    def label(self, label):
        """Sets the label of this Experiment.


        :param label: The label of this Experiment.
        :type label: str
        """

        self._label = label

    @property
    def version(self):
        """Gets the version of this Experiment.


        :return: The version of this Experiment.
        :rtype: str
        """
        return self._version

    @version.setter
    def version(self, version):
        """Sets the version of this Experiment.


        :param version: The version of this Experiment.
        :type version: str
        """

        self._version = version

    @property
    def train_events(self):
        """Gets the train_events of this Experiment.


        :return: The train_events of this Experiment.
        :rtype: List[Result]
        """
        return self._train_events

    @train_events.setter
    def train_events(self, train_events):
        """Sets the train_events of this Experiment.


        :param train_events: The train_events of this Experiment.
        :type train_events: List[Result]
        """

        self._train_events = train_events

    @property
    def valid_events(self):
        """Gets the valid_events of this Experiment.


        :return: The valid_events of this Experiment.
        :rtype: List[Result]
        """
        return self._valid_events

    @valid_events.setter
    def valid_events(self, valid_events):
        """Sets the valid_events of this Experiment.


        :param valid_events: The valid_events of this Experiment.
        :type valid_events: List[Result]
        """

        self._valid_events = valid_events

    @property
    def test_events(self):
        """Gets the test_events of this Experiment.


        :return: The test_events of this Experiment.
        :rtype: List[Result]
        """
        return self._test_events

    @test_events.setter
    def test_events(self, test_events):
        """Sets the test_events of this Experiment.


        :param test_events: The test_events of this Experiment.
        :type test_events: List[Result]
        """

        self._test_events = test_events

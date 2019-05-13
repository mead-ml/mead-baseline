# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from xpctl.server.models.base_model_ import Model
from xpctl.server import util


class TaskSummary(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(self, task=None, summary=None):  # noqa: E501
        """TaskSummary - a model defined in Swagger

        :param task: The task of this TaskSummary.  # noqa: E501
        :type task: str
        :param summary: The summary of this TaskSummary.  # noqa: E501
        :type summary: object
        """
        self.swagger_types = {
            'task': str,
            'summary': object
        }

        self.attribute_map = {
            'task': 'task',
            'summary': 'summary'
        }

        self._task = task
        self._summary = summary

    @classmethod
    def from_dict(cls, dikt):
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The TaskSummary of this TaskSummary.  # noqa: E501
        :rtype: TaskSummary
        """
        return util.deserialize_model(dikt, cls)

    @property
    def task(self):
        """Gets the task of this TaskSummary.


        :return: The task of this TaskSummary.
        :rtype: str
        """
        return self._task

    @task.setter
    def task(self, task):
        """Sets the task of this TaskSummary.


        :param task: The task of this TaskSummary.
        :type task: str
        """
        if task is None:
            raise ValueError("Invalid value for `task`, must not be `None`")  # noqa: E501

        self._task = task

    @property
    def summary(self):
        """Gets the summary of this TaskSummary.


        :return: The summary of this TaskSummary.
        :rtype: object
        """
        return self._summary

    @summary.setter
    def summary(self, summary):
        """Sets the summary of this TaskSummary.


        :param summary: The summary of this TaskSummary.
        :type summary: object
        """
        if summary is None:
            raise ValueError("Invalid value for `summary`, must not be `None`")  # noqa: E501

        self._summary = summary

# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.error import Error  # noqa: E501
from swagger_server.models.experiment import Experiment  # noqa: E501
from swagger_server.models.experiment_aggregate import ExperimentAggregate  # noqa: E501
from swagger_server.test import BaseTestCase


class TestXpctlController(BaseTestCase):
    """XpctlController integration test stubs"""

    def test_experiment_details(self):
        """Test case for experiment_details

        Find experiment by id
        """
        response = self.client.open(
            '/v2/{task}/{eid}'.format(task='task_example', eid='eid_example'),
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_find_by_label(self):
        """Test case for find_by_label

        Finds experiment by label
        """
        query_string = [('label', 'label_example')]
        response = self.client.open(
            '/v2/findbylabel',
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_results(self):
        """Test case for get_results

        Find results by dataset and task
        """
        query_string = [('task', 'task_example'),
                        ('dataset', 'dataset_example'),
                        ('metric', 'metric_example'),
                        ('sort', 'sort_example'),
                        ('nconfig', 56),
                        ('event_type', 'event_type_example')]
        response = self.client.open(
            '/v2/results',
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_put_result(self):
        """Test case for put_result

        Add a new experiment in database
        """
        experiment = Experiment()
        response = self.client.open(
            '/v2/putresult',
            method='POST',
            data=json.dumps(experiment),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_remove_experiment(self):
        """Test case for remove_experiment

        Deletes an experiment
        """
        response = self.client.open(
            '/v2/{task}/{eid}'.format(task='task_example', eid='eid_example'),
            method='DELETE')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_update_label(self):
        """Test case for update_label

        Updates an experiment in the database with form data
        """
        data = dict(label='label_example')
        response = self.client.open(
            '/v2/{task}/{eid}'.format(task='task_example', eid='eid_example'),
            method='POST',
            data=data,
            content_type='application/x-www-form-urlencoded')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()

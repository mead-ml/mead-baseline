# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.experiment import Experiment  # noqa: E501
from swagger_server.models.experiment_aggregate import ExperimentAggregate  # noqa: E501
from swagger_server.models.response import Response  # noqa: E501
from swagger_server.models.task_summary import TaskSummary  # noqa: E501
from swagger_server.test import BaseTestCase


class TestXpctlController(BaseTestCase):
    """XpctlController integration test stubs"""

    def test_config2json(self):
        """Test case for config2json

        get config for sha1
        """
        response = self.client.open(
            '/v2/config2json/{task}/{sha1}'.format(task='task_example', sha1='sha1_example'),
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_experiment_details(self):
        """Test case for experiment_details

        Find experiment by id
        """
        response = self.client.open(
            '/v2/{task}/{eid}'.format(task='task_example', eid='eid_example'),
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_model_location(self):
        """Test case for get_model_location

        get model loc for experiment
        """
        response = self.client.open(
            '/v2/getmodelloc/{task}/{eid}'.format(task='task_example', eid='eid_example'),
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_results_by_dataset(self):
        """Test case for get_results_by_dataset

        Find results by dataset and task
        """
        query_string = [('reduction_dim', 'reduction_dim_example'),
                        ('metric', 'metric_example'),
                        ('sort', 'sort_example'),
                        ('numexp_reduction_dim', 56),
                        ('event_type', 'event_type_example')]
        response = self.client.open(
            '/v2/results/{task}/{dataset}'.format(task='task_example', dataset='dataset_example'),
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_results_by_prop(self):
        """Test case for get_results_by_prop

        Find results by property and value
        """
        query_string = [('prop', 'prop_example'),
                        ('value', 'value_example'),
                        ('reduction_dim', 'reduction_dim_example'),
                        ('metric', 'metric_example'),
                        ('sort', 'sort_example'),
                        ('numexp_reduction_dim', 56),
                        ('event_type', 'event_type_example')]
        response = self.client.open(
            '/v2/results/{task}'.format(task='task_example'),
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_list_experiments_by_prop(self):
        """Test case for list_experiments_by_prop

        list all experiments for this property and value
        """
        query_string = [('prop', 'prop_example'),
                        ('value', 'value_example'),
                        ('user', 'user_example'),
                        ('metric', 'metric_example'),
                        ('sort', 'sort_example'),
                        ('event_type', 'event_type_example')]
        response = self.client.open(
            '/v2/details/{task}'.format(task='task_example'),
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_list_experiments_by_sha1(self):
        """Test case for list_experiments_by_sha1

        list all experiments for this sha1
        """
        query_string = [('user', 'user_example'),
                        ('metric', 'metric_example'),
                        ('sort', 'sort_example'),
                        ('event_type', 'event_type_example')]
        response = self.client.open(
            '/v2/details/{task}/{sha1}'.format(task='task_example', sha1='sha1_example'),
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_put_result(self):
        """Test case for put_result

        Add a new experiment in database
        """
        experiment = Experiment()
        query_string = [('user', 'user_example'),
                        ('label', 'label_example')]
        response = self.client.open(
            '/v2/put/{task}'.format(task='task_example'),
            method='POST',
            data=json.dumps(experiment),
            content_type='application/json',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_remove_experiment(self):
        """Test case for remove_experiment

        delete an experiment from the database
        """
        response = self.client.open(
            '/v2/delete/{task}/{eid}'.format(task='task_example', eid='eid_example'),
            method='GET',
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_summary(self):
        """Test case for summary

        get summary for task
        """
        response = self.client.open(
            '/v2/summary/',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_task_summary(self):
        """Test case for task_summary

        get summary for task
        """
        response = self.client.open(
            '/v2/summary/{task}/'.format(task='task_example'),
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_update_label(self):
        """Test case for update_label

        update label for an experiment
        """
        query_string = [('label', 'label_example')]
        response = self.client.open(
            '/v2/updatelabel/{task}/{eid}/'.format(task='task_example', eid='eid_example'),
            method='GET',
            content_type='application/json',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()

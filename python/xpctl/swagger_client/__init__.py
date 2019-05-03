# coding: utf-8

# flake8: noqa

"""
    xpctl

    This is a sample xpctl  server.  You can find out more about xpctl at [baseline](https://github.com/dpressel/baseline/blob/master/docs/xpctl.md).  # noqa: E501

    OpenAPI spec version: 1.0.0
    Contact: apiteam@swagger.io
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


from __future__ import absolute_import

# import apis into sdk package
from swagger_client.api.xpctl_api import XpctlApi

# import ApiClient
from swagger_client.api_client import ApiClient
from swagger_client.configuration import Configuration
# import models into sdk package
from swagger_client.models.aggregate_result import AggregateResult
from swagger_client.models.error import Error
from swagger_client.models.experiment import Experiment
from swagger_client.models.experiment_aggregate import ExperimentAggregate
from swagger_client.models.result import Result

from __future__ import print_function
from baseline.utils import export

__all__ = []
exporter = export(__all__)

EVENT_TYPES = {
    "train": "train_events", "Train": "train_events",
    "test": "test_events", "Test": "test_events",
    "valid": "valid_events", "Valid": "valid_events",
    "dev": "valid_events", "Dev": "valid_events"
}


@exporter
class ExperimentRepo(object):

    def __init__(self):
        super(ExperimentRepo, self).__init__()

    def get_task_names(self):
        """Get the names of all tasks in the repository

        :return: A list of tasks
        """
        pass

    def config2dict(self, task, sha1):
        """Convert a configuration stored in the repository to a string

        :param task: (``str``) The task name
        :param sha1: (``str``) The sha1 of the configuration
        :return: (``dict``) The configuration
        """
        pass

    @staticmethod
    def create_repo(dbhost, dbport, user, passwd, dbtype='mongo'):
        """Create a MongoDB-backed repository

        :param dbtype: (``str``) The database type
        :param dbhost: (``str``) The host name
        :param dbport: (``str``) The port
        :param user: (``str``) The user
        :param passwd: (``str``)The password
        :return: A MongoDB/Sql-backed repository
        """
        if dbtype == 'mongo':
            from xpctl.backends.mongo.store import MongoRepo
            return MongoRepo(dbhost, int(dbport), user, passwd)
        else:
            from xpctl.backends.sql.store import SQLRepo
            return SQLRepo(type=dbtype, host=dbhost, port=dbport, user=user, passwd=passwd)

    def get_model_location(self, task, eid):
        """Get the physical location of the model specified by this experiment id

        :param task: (``str``) The task name
        :param eid: The identifier of the experiment
        :return: (``str``) The model location
        """
        pass

    def get_experiment_details(self, task, eid, event_type, metric):
        """Get detailed description for an experiment
        :param task: (``str``) The task name
        :param eid: The identifier of the experiment
        :param event_type: (List[``str``]) event types to listen for
        :param metric: (List[``str``]) The metric(s) to use
        :return: xpctl.backend.data.Experiment object
        """
        pass
        
    def get_results(self, task, prop, value, reduction_dim, metric, sort, numexp_reduction_dim, event_type):
        """Get results from the database
        :param task: (``str``) The taskname
        :param prop: (``str``) Filter the experiments by a property(eg: 'dataset')
        :param value: (``str``) The value for the property (eg: 'SST2')
        :param reduction_dim: (``str``) The property on which the results will be reduced
        :param metric: (``str``) The metric(s) to use
        :param sort: (``str``) The field to sort on
        :param numexp_reduction_dim: (``str``) number of results to show per group
        :param event_type: (``str``) event types to listen for
        :return: List[xpctl.backend.data.ExperimentAggregate]
        """
        pass

    def task_summary(self, task):
        """Summary for a task: What datasets were used? How many times each dataset was used?
        :param task: (``str``) Task name
        :return xpctl.backend.data.TaskSummary
        """
        pass

    def summary(self):
        """
        Summary for all tasks in the database
        :return: List[xpctl.backend.data.TaskSummary]
        """
        pass

    def find_experiments(self, task, prop, value):
        """
        find experiments by a property, eg: dataset
        :param task: (``str``) task name
        :param prop: (``str``) a property of an experiment, eg: dataset, label
        :param value: (``str``) value for the property
        :return: List[xpctl.backend.data.Experiment]
        """
        pass
    
    def update_prop(self, task, eid, prop, value):
        """
        Update a property(label, username etc) for an experiment
        :param task: (``str``) task name
        :param eid: The identifier for this record
        :param prop: (``str``) property to change
        :param value: (``str``) the value of the property
        :return: Union[xpctl.backend.data.Success, xpctl.backend.data.Error]
        """
        raise NotImplemented("Base ExperimentRepo events are immutable")

    def remove_experiment(self, task, eid):
        """Remove a record specified by this id
        :param task: (``str``) The task name for this record
        :param eid: The identifier for this record
        :return: Union[xpctl.backend.data.Success, xpctl.backend.data.Error]
        """
        raise NotImplemented("Base ExperimentRepo tasks are immutable")

    def put_result(self, task, experiment):
        """Put tan experiment in the database

        :param task: (``str``) The task name
        :param experiment: xpctl.backend.data.Experiment
        :return: Union[xpctl.backend.data.Success, xpctl.backend.data.Error]
        """
        pass

    def list_results(self, task, prop, value, user, metric, sort, event_type):
        """Get results from the database
        :param task: (``str``) The taskname
        :param prop: (``str``) List the experiments by a property(eg: 'dataset')
        :param value: (``str``) The value for the property (eg: 'SST2')
        :param user: List[``str``)] filter by users
        :param metric: (``str``) The metric(s) to use
        :param sort: (``str``) The field to sort on
        :param event_type: (``str``) event types to listen for
        :return: List[xpctl.backend.data.Experiment]
        """
        pass

    def dump(self, zipfile, task_eids):
        """
        dump the whole database. creates a zipfile which unzips into the following directory structure
        <xpctldump>
         - <task>
           - <id>
             - <id>-reporting.log
             - <id>-config.yml
             - <id>-meta.yml (any meta info such as label, username etc.)
        
        :param zipfile: zip file location for the dump. defaults to xpctldump-datetimestamp.zip
        :param task_eids: a dictionary of the form {'task': [eid1, eid2]}, if you want to dump specific files.
        :return: the path to the dumped file
        """
        pass
    
    def restore(self, dump):
        """
        restore a database from the dump. be careful: the experiment ids will change.
        :param dump: dump file location
        :return:
        """
        pass

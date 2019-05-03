from __future__ import print_function
import os
import shutil
from baseline.utils import export, unzip_model

__all__ = []
exporter = export(__all__)

EVENT_TYPES = {
    "train": "train_events", "Train": "train_events",
    "test": "test_events", "Test": "test_events",
    "valid": "valid_events", "Valid": "valid_events",
    "dev": "valid_events", "Dev": "valid_events"
}


@exporter
def store_model(checkpoint_base, config_sha1, checkpoint_store, print_fn=print):
    checkpoint_base = unzip_model(checkpoint_base)
    mdir, mbase = os.path.split(checkpoint_base)
    mdir = mdir if mdir else "."
    if not os.path.exists(mdir):
        print_fn("no directory found for the model location: [{}], aborting command".format(mdir))
        return None

    mfiles = ["{}/{}".format(mdir, x) for x in os.listdir(mdir) if x.startswith(mbase + "-") or
              x.startswith(mbase + ".")]
    if not mfiles:
        print_fn("no model files found with base [{}] at location [{}], aborting command".format(mbase, mdir))
        return None
    model_loc_base = "{}/{}".format(checkpoint_store, config_sha1)
    if not os.path.exists(model_loc_base):
        os.makedirs(model_loc_base)
    dirs = [int(x[:-4]) for x in os.listdir(model_loc_base) if x.endswith(".zip") and x[:-4].isdigit()]
    # we expect dirs in numbers.
    new_dir = "1" if not dirs else str(max(dirs) + 1)
    model_loc = "{}/{}".format(model_loc_base, new_dir)
    os.makedirs(model_loc)
    for mfile in mfiles:
        shutil.copy(mfile, model_loc)
        print_fn("writing model file: [{}] to store: [{}]".format(mfile, model_loc))
    print_fn("zipping model files")
    shutil.make_archive(base_name=model_loc,
                        format='zip',
                        root_dir=model_loc_base,
                        base_dir=new_dir)
    shutil.rmtree(model_loc)
    print_fn("model files zipped and written")
    return model_loc + ".zip"


@exporter
class ExperimentRepo(object):

    def __init__(self):
        super(ExperimentRepo, self).__init__()

    def get_task_names(self):
        """Get the names of all tasks in the repository

        :return: A list of tasks
        """
        pass

    def has_task(self, task):
        """Does this task exist in the repository

        :param task: (``str``) Task name
        :return: (``bool``) `True` if it exist, `False` if it does not
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

        :param type: (``str``) The database type
        :param host: (``str``) The host name
        :param port: (``str``) The port
        :param user: (``str``) The user
        :param passw: (``str``)The password
        :return: A MongoDB-backed repository
        """
        if dbtype == 'mongo':
            from xpctl.backend.mongo.backend import MongoRepo
            return MongoRepo(dbhost, int(dbport), user, passwd)
        else:
            from xpctl.backend.sql.backend import SQLRepo
            return SQLRepo(type=dbtype, host=dbhost, port=dbport, user=user, passwd=passwd)

    def get_model_location(self, id, task):
        """Get the physical location of the model specified by this id

        :param id: The identifier of the run
        :param task: (``str``) The task name
        :return: (``str``) The model location
        """
        pass

    def get_results(self, task, dataset, event_type, num_exps, num_exps_per_config, metric, sort, id, label):
        """Get results from the database

        :param task: (``str``) The taskname
        :param dataset: (``str``) The dataset
        :param event_type: (``str``) event types to listen for
        :param num_exps: (``str``) number of results to show
        :param num_exps_per_config: (``str``) number of results to show per config
        :param event_type: (``str``) event types to listen for
        :param metric: (``str``) The metric to use
        :param sort: (``str``) The field to sort on
        :return: A result DataFrame
        """
        pass

    def get_info(self, task, event_types):
        """Show datasets that are available for this task, what metrics and which username and hostnames

        :param task: (``str``) Task name
        :param event_types: (``list``) A list of ``str`` of event types
        """
        pass

    def leaderboard_summary(self, task=None, event_type=None, print_fn=print):
        pass

    def get_label(self, id, task):
        """Get the label for the record with this id

        :param id: The identifier for this record
        :param task: (``str``) The task name
        :return: (``str``) Return the user-defined label for this task
        """
        pass

    def rename_label(self, id, task, new_label):
        """Rename the user-defined label for the task identified by this id

        :param id: The identifier for this record
        :param task: (``str``) The task name
        :param new_label: (``str``) The new label
        :return: (``tuple``) A tuple of the old name and then the new name
        """
        raise NotImplemented("Base ExperimentRepo events are immutable")

    def rm(self, id, task, print_fn=print):
        """Remove a record specified by this id

        :param id: The identifier for this record
        :param task: (``str``) The task name for this record
        :param print_fn: A print callback which takes a ``str`` argument
        :return: (``bool``) True if something was removed
        """
        raise NotImplemented("Base ExperimentRepo tasks are immutable")

    def put_model(self, id, task, checkpoint_base, checkpoint_store, print_fn=print):
        """Put the model for the record identified by id to the the checkpoint store

        :param id:  The identifier
        :param task: (``str``) The task name
        :param checkpoint_base: (``str``) The basename of the model
        :param checkpoint_store: (``str``) A path to the checkpoint store
        :param print_fn: A print callback which takes a ``str`` argument
        :return: The model location
        """
        pass

    def put_result(self, task, config_obj, events_obj, **kwargs):
        """Put the result to the experiment repository

        :param task: (``str``) The task name
        :param config_obj: (``dict``) A dictionary containing the job configuration
        :param events_obj: (``dict``) A dictionary containing the events that transpired during training
        :param kwargs: See below

        label = kwargs.get("label", id)

        :Keyword Arguments:
        * *checkpoint_base* (``str``) -- If we are putting the model simultaneously, required basename for model
        * *checkpoint_store* (``str``) -- If we are putting the model simultaneously, required destination
        * *print_fn* -- A print callback which takes a ``str`` argument
        * *hostname* -- (``str``) A hostname, defaults to name of the local machine
        * *username* -- (``str``) A username, defaults to the name of the user on this machine
        * *label* -- (``str``) An optional, human-readable label name.  Defaults to sha1 of this configuration

        :return: The identifier assigned by the database
        """
        pass

    def experiment_details(self, user, metric, sort, task, event_type, sha1, n):
        """Get results from the database

        :param user: (``str``) The username
        :param metric: (``str``) The metric to use
        :param sort: (``str``) The field to sort on
        :param task: (``str``) The task
        :param event_type: (``str``) event types to listen for
        :param sha1: (``str``) sha1 of the config for the experiment
        :param n: (``int``) number of results to show.
        :return: A result DataFrame
        """
        pass

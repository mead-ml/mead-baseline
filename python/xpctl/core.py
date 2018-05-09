from __future__ import print_function
import os
import shutil
import pandas as pd
import pymongo
import datetime
import socket
import json
import hashlib
import getpass
from baseline.utils import export, listify
from bson.objectid import ObjectId
from baseline.version import __version__

__all__ = []
exporter = export(__all__)


@exporter
def store_model(checkpoint_base, config_sha1, checkpoint_store, print_fn=print):
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
    print_fn("zipped file written, model directory removed")
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

    def nbest_by_metric(self, username, metric, dataset, task, num_results, event_type, ascending):
        """Get the n-best results according to the specific metrics

        :param username: (``str``) Name of user or None
        :param metric: (``str``) The name of the metric to use for N-best
        :param dataset: (``str``) The name of the dataset
        :param task: (``str``) The name of the task
        :param num_results: (``int``) The number of results (max) to retrieve from the repo
        :param event_type: (``str``) The event types to retrieve from repo
        :param ascending: (``bool``) Should we sort ascending or descending (depends on metric)
        :return: The N-best data frame
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
    def create_mongo_repo(host, port, user, passw):
        """Create a MongoDB-backed repository

        :param host: (``str``)The host name
        :param port: (``str``)The port
        :param user: (``str``)The user
        :param passw: (``str``)The password
        :return: A MongoDB-backed repository
        """
        return MongoRepo(host, port, user, passw)

    def get_model_location(self, id, task):
        """Get the physical location of the model specified by this id

        :param id: The identifier of the run
        :param task: (``str``) The task name
        :return: (``str``) The model location
        """
        pass

    def get_results(self, username, metric, sort, dataset, task, event_type):
        """Get results from the database

        :param username: (``str``) The username
        :param metric: (``str``) The metric to use
        :param sort: (``str``) The field to sort on
        :param dataset: (``str``) The dataset
        :param task: (``str``) The task
        :param event_type: (``str``) event types to listen for
        :return: A result DataFrame
        """
        pass

    def get_info(self, task, event_types):
        """Show datasets that are available for this task, what metrics and which username and hostnames

        :param task: (``str``) Task name
        :param event_types: (``list``) A list of ``str`` of event types
        """
        pass

    def leaderboard_summary(self, task=None, event_types=None, print_fn=print):
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


@exporter
class MongoRepo(ExperimentRepo):

    def __init__(self, host, port, user, passw):
        super(MongoRepo, self).__init__()
        self.dbhost = host
        if user and passw:
            uri = "mongodb://{}:{}@{}:{}/test".format(user, passw, host, port)
            client = pymongo.MongoClient(uri)
        else:
            client = pymongo.MongoClient(host, port)
        if client is None:
            s = "can not connect to mongo at host: [{}], port [{}], username: [{}], password: [{}]".format(host,
                                                                                                           port,
                                                                                                           user,
                                                                                                           passw)
            raise Exception(s)
        try:
            dbnames = client.database_names()
        except pymongo.errors.ServerSelectionTimeoutError:
            raise Exception("can not get database from mongo at host: {}, port {}, connection timed out".format(host,
                                                                                                                port))

        if "reporting_db" not in dbnames:
            raise Exception("no database for results found")
        self.db = client.reporting_db

    def put_result(self, task, config_obj, events_obj, **kwargs):
        now = datetime.datetime.utcnow().isoformat()
        train_events = list(filter(lambda x: x['phase'] == 'Train', events_obj))
        valid_events = list(filter(lambda x: x['phase'] == 'Valid', events_obj))
        test_events = list(filter(lambda x: x['phase'] == 'Test', events_obj))

        checkpoint_base = kwargs.get('checkpoint_base', None)
        checkpoint_store = kwargs.get('checkpoint_store', None)
        print_fn = kwargs.get('print_fn', print)
        hostname = kwargs.get('hostname', socket.gethostname())
        username = kwargs.get('username', getpass.getuser())
        config_sha1 = hashlib.sha1(json.dumps(config_obj).encode('utf-8')).hexdigest()
        label = kwargs.get("label", config_sha1)

        post = {
            "config": config_obj,
            "train_events": train_events,
            "valid_events": valid_events,
            "test_events": test_events,
            "username": username,
            "hostname": hostname,
            "date": now,
            "label": label,
            "sha1": config_sha1,
            "version": __version__
        }

        if checkpoint_base:
            model_loc = store_model(checkpoint_base, config_sha1, checkpoint_store)
            if model_loc is not None:
                post.update({"checkpoint": "{}:{}".format(hostname, os.path.abspath(model_loc))})
            else:
                print_fn("model could not be stored, see previous errors")

        if task in self.db.collection_names():
            print_fn("updating results for existing task [{}] in host [{}]".format(task, self.dbhost))
        else:
            print_fn("creating new task [{}] in host [{}]".format(task, self.dbhost))
        coll = self.db[task]
        result = coll.insert_one(post)

        print_fn("results updated, the new results are stored with the record id: {}".format(result.inserted_id))
        return result.inserted_id

    def has_task(self, task):
        return task in self.get_task_names()

    def put_model(self, id, task, checkpoint_base, checkpoint_store, print_fn=print):
        coll = self.db[task]
        query = {'_id': ObjectId(id)}
        projection = {'sha1': 1}
        results = list(coll.find(query, projection))
        if not results:
            print_fn("no sha1 for the given id found, returning.")
            return False
        sha1 = results[0]['sha1']
        model_loc = store_model(checkpoint_base, sha1, checkpoint_store, print_fn)
        if model_loc is not None:
            coll.update_one({'_id': ObjectId(id)}, {'$set': {'checkpoint': model_loc}}, upsert=False)
        return model_loc

    def get_label(self, id, task):
        coll = self.db[task]
        label = coll.find_one({'_id': ObjectId(id)}, {'label': 1})["label"]
        return label

    def rename_label(self, id, task, new_label):
        coll = self.db[task]
        prev_label = coll.find_one({'_id': ObjectId(id)}, {'label': 1})["label"]
        coll.update({'_id': ObjectId(id)}, {'$set': {'label': new_label}}, upsert=False)
        changed_label = coll.find_one({'_id': ObjectId(id)}, {'label': 1})["label"]
        return prev_label, changed_label

    def rm(self, id, task, print_fn=print):
        coll = self.db[task]
        prev = coll.find_one({'_id': ObjectId(id)}, {'label': 1})
        if prev is None:
            return False

        model_loc = self.get_model_location(id, task)
        if model_loc is not None and os.path.exists(model_loc):
                os.remove(model_loc)
        else:
            print_fn("No model stored for this record. Only purging the database.")
        coll.remove({'_id': ObjectId(id)})
        assert coll.find_one({'_id': ObjectId(id)}) is None
        print_fn("record {} deleted successfully from database {}".format(id, task))
        return True

    def _get_metrics(self, xs, event_type):
        keys = []
        for x in xs:
            if x[event_type]:
                for k in x[event_type][0].keys():
                    keys.append(k)
        keys = set(keys)
        if 'tick_type' in keys:
            keys.remove("tick_type")
        if 'tick' in keys:
            keys.remove("tick")
        if 'phase' in keys:
            keys.remove("phase")
        return keys

    def _generate_data_frame(self, coll, metrics, query, projection, event_type):
        all_results = list(coll.find(query, projection))
        if not all_results:
            return pd.DataFrame()

        results = []
        ms = list(set(metrics)) if len(metrics) > 0 else list(self._get_metrics(all_results, event_type))
        for result in all_results:  # different experiments
            for index in range(len(result[event_type])):  # train_event epoch 0,
                # train_event epoch 1 etc, for event_type = test_event, there is only one event
                data = []
                for metric in ms:
                    data.append(result[event_type][index][metric])
                results.append(
                    [result['_id'], result['username'], result['label'], result['config']['dataset'], result['sha1'],
                     result['date']] + data)
        return pd.DataFrame(results, columns=['id', 'username', 'label', 'dataset', 'sha1', 'date'] + ms)

    def _update_query(self, q, uname, dataset):
        query = q
        if uname:
            query.update({"username": {"$in": list(uname)}})
        if dataset:
            query.update({"config.dataset": dataset})
        return query

    def _update_projection(self, event_type):
        projection = {"_id": 1, "sha1": 1, "label": 1, "username": 1, "config.dataset": 1, "date": 1}
        projection.update({event_type: 1})
        return projection

    def nbest_by_metric(self, username, metric, dataset, task, num_results, event_type, ascending):
        metrics = listify(metric)
        coll = self.db[task]
        query = self._update_query({}, username, dataset)
        projection = self._update_projection(event_type)
        result_frame = self._generate_data_frame(coll, metrics, query, projection, event_type)
        if not result_frame.empty:
            return result_frame.sort_values(metrics,
                                            ascending=[ascending])[:min(int(num_results), result_frame.shape[0])]
        return None

    def get_results(self, username, metric, sort, dataset, task, event_type):
        event_type = event_type.lower()
        metrics = list(metric)
        coll = self.db[task]
        query = self._update_query({}, username, dataset)
        projection = self._update_projection(event_type=event_type)
        result_frame = self._generate_data_frame(coll, metrics, query, projection, event_type=event_type)
        if len(metric) == 1:
            metric = metric[0]
            if metric == "avg_loss" or metric == "perplexity":
                result_frame = result_frame.sort_values(metric, ascending=True)
            else:
                result_frame = result_frame.sort_values(metric, ascending=False)
            if sort:
                if sort == "avg_loss" or sort == "perplexity":
                    result_frame = result_frame.sort_values(sort, ascending=True)
                else:
                    result_frame = result_frame.sort_values(sort, ascending=False)

        if not result_frame.empty:
            return result_frame
        return None

    def task_summary(self, task, dataset, metric, event_type):
        metrics = listify(metric)

        coll = self.db[task]
        query = self._update_query({}, [], dataset)
        projection = self._update_projection(event_type=event_type)
        result_frame = self._generate_data_frame(coll, metrics, query, projection, event_type=event_type)
        if not result_frame.empty:
            datasets = result_frame.dataset.unique()
            if dataset not in datasets:
                return None
            dsr = result_frame[result_frame.dataset == dataset].sort_values(metric, ascending=False)
            result = dsr[metric].iloc[0]
            user = dsr.username.iloc[0]
            sha1 = dsr.sha1.iloc[0]
            date = dsr.date.iloc[0]
            summary = "For dataset {}, the best {} is {:0.3f} reported by {} on {}. " \
                      "The sha1 for the config file is {}.".format(dataset, metric, result, user, date, sha1)
            return summary

        return None

    def config2dict(self, task, sha1):
        coll = self.db[task]
        j = coll.find_one({"sha1": sha1}, {"config": 1})["config"]
        if not j:
            return None
        else:
            return j

    def get_task_names(self):
        return self.db.collection_names()

    def get_model_location(self, id, task):
        coll = self.db[task]
        query = {'_id': ObjectId(id)}
        projection = {'checkpoint': 1}
        results = [x.get('checkpoint', None) for x in list(coll.find(query, projection))]
        results = [x for x in results if x]
        if not results:
            return None
        return results[0]

    def get_info(self, task, event_types):
        coll = self.db[task]
        q = self._update_query({}, None, None)
        p = {'config.dataset': 1}
        datasets = list(set([x['config']['dataset'] for x in list(coll.find(q, p))]))
        store = []  #

        for dataset in datasets:
            q = self._update_query({}, None, dataset)
            for event_type in event_types:
                p = self._update_projection(event_type)
                results = list(coll.find(q, p))
                metrics = self._get_metrics(results, event_type)
                for result in results:  # different experiments
                    store.append([result['username'], result['config']['dataset'], event_type, ",".join(metrics)])

        df = pd.DataFrame(store, columns=['user', 'dataset', 'event_type', 'metrics'])
        return df.groupby(['user', 'dataset', 'event_type', 'metrics']).size().reset_index() \
            .rename(columns={0: 'num_experiments'})

    def leaderboard_summary(self, task=None, event_types=None, print_fn=print):

        if task:
            print_fn("Task: [{}]".format(task))
            print_fn("-" * 93)
            print_fn(self.get_info(task, event_types))
        else:
            tasks = self.db.collection_names()
            if "system.indexes" in tasks:
                tasks.remove("system.indexes")
            print_fn("There are {} tasks: {}".format(len(tasks), tasks))
            for task in tasks:
                print_fn("-" * 93)
                print_fn("Task: [{}]".format(task))
                print_fn("-" * 93)
                print_fn(self.get_info(task, event_types))


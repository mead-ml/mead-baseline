from __future__ import print_function
import os
import pandas as pd
import pymongo
import datetime
import socket
import json
import hashlib
import getpass
from baseline.utils import export, listify
from xpctl.core import ExperimentRepo, store_model
from bson.objectid import ObjectId
from baseline.version import __version__
from xpctl.helpers import order_json, df_get_results, df_experimental_details, get_experiment_label

__all__ = []
exporter = export(__all__)


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
        config_sha1 = hashlib.sha1(json.dumps(order_json(config_obj)).encode('utf-8')).hexdigest()
        label = get_experiment_label(config_obj, task, **kwargs)

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
                    [result['_id'], result['username'], result['label'], result['config']['dataset'], result.get('sha1'),
                     result['date']] + data)
        return pd.DataFrame(results, columns=['id', 'username', 'label', 'dataset', 'sha1', 'date'] + ms)

    def _update_query(self, q, **kwargs):
        query = q
        if not kwargs:
            return query
        else:
            if "username" in kwargs and kwargs["username"]:
                query.update({"username": {"$in": list(kwargs["username"])}})
            if "dataset" in kwargs:
                query.update({"config.dataset": kwargs["dataset"]})
            if "sha1" in kwargs:
                query.update({"sha1": kwargs["sha1"]})
            return query

    def _update_projection(self, event_type):
        projection = {"_id": 1, "sha1": 1, "label": 1, "username": 1, "config.dataset": 1, "date": 1}
        projection.update({event_type: 1})
        return projection

    def experiment_details(self, user, metric, sort, task, event_type, sha1, n):
        metrics = listify(metric)
        coll = self.db[task]
        users = listify(user)
        query = self._update_query({}, username=users, sha1=sha1)
        projection = self._update_projection(event_type=event_type)
        result_frame = self._generate_data_frame(coll, metrics=metrics, query=query, projection=projection, event_type=event_type)
        return df_experimental_details(result_frame, sha1, users, sort, metric, n)

    def get_results(self, task, dataset, event_type,  num_exps=None, num_exps_per_config=None, metric=None, sort=None):
        metrics = listify(metric)
        coll = self.db[task]
        query = self._update_query({}, dataset=dataset)
        projection = self._update_projection(event_type=event_type)
        result_frame = self._generate_data_frame(coll, metrics=metrics, query=query, projection=projection, event_type=event_type)
        if not result_frame.empty:
            return df_get_results(result_frame, dataset, num_exps, num_exps_per_config, metric, sort)
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

    def get_info(self, task, event_type):
        coll = self.db[task]
        q = {}
        p = {'config.dataset': 1}
        datasets = list(set([x['config']['dataset'] for x in list(coll.find(q, p))]))
        store = []
        for dataset in datasets:
            q = self._update_query({}, dataset=dataset)
            p = self._update_projection(event_type)
            results = list(coll.find(q, p))
            for result in results:  # different experiments
                store.append([result['username'], result['config']['dataset'], task])

        df = pd.DataFrame(store, columns=['user', 'dataset', 'task'])
        return df.groupby(['user', 'dataset']).agg([len]) \
            .rename(columns={"len": 'num_exps'})

    def leaderboard_summary(self, event_type, task=None, print_fn=print):
        if task:
            print_fn("Task: [{}]".format(task))
            print_fn("-" * 93)
            print_fn(self.get_info(task, event_type))
        else:
            tasks = self.db.collection_names()
            if "system.indexes" in tasks:
                tasks.remove("system.indexes")
            print_fn("There are {} tasks: {}".format(len(tasks), tasks))
            for task in tasks:
                print_fn("-" * 93)
                print_fn("Task: [{}]".format(task))
                print_fn("-" * 93)
                print_fn(self.get_info(task, event_type))


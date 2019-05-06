from __future__ import print_function
import os
import shutil
import pymongo
import datetime
import socket
import getpass
from baseline.utils import export, listify, read_config_file, write_config_file, unzip_files
from mead.utils import hash_config, configure_logger
from xpctl.backend.core import ExperimentRepo, store_model, EVENT_TYPES
from xpctl.backend.mongo.dto import MongoResult, MongoResultSet, MongoError, MongoTaskDatasetSummary, MongoTaskDatasetSummarySet
from xpctl.backend.helpers import log2json, get_experiment_label, METRICS_SORT_ASCENDING
from bson.objectid import ObjectId
from baseline.version import __version__
import numpy as np
import json
import logging


__all__ = []
exporter = export(__all__)


@exporter
class MongoRepo(ExperimentRepo):

    def __init__(self, dbhost, dbport, user, passwd):
        super(MongoRepo, self).__init__()
        self.logger = logging.getLogger('xpctl-mongo')
        self.dbhost = dbhost
        if user and passwd:
            uri = "mongodb://{}:{}@{}:{}/test".format(user, passwd, dbhost, dbport)
            client = pymongo.MongoClient(uri)
        else:
            client = pymongo.MongoClient(dbhost, dbport)
        if client is None:
            s = "cannot connect to mongo at host: [{}], port [{}], username: [{}], password: [{}]".format(dbhost,
                                                                                                          dbport,
                                                                                                          user,
                                                                                                          passwd)
            raise Exception(s)
        try:
            dbnames = client.list_database_names()
        except pymongo.errors.ServerSelectionTimeoutError:
            raise Exception("cannot get database from mongo at host: {}, port {}, connection timed out".format(dbhost,
                                                                                                               dbport))

        if "reporting_db" not in dbnames:
            self.logger.warning("Warning: database reporting_db does not exist, do not query before inserting results")
        self.db = client.reporting_db

    @staticmethod
    def get_checkpoint(checkpoint_base, checkpoint_store, config_sha1, hostname):
        if checkpoint_base:
            model_loc = store_model(checkpoint_base, config_sha1, checkpoint_store)
            if model_loc is not None:
                return "{}:{}".format(hostname, os.path.abspath(model_loc))
            else:
                raise RuntimeError("model could not be stored, see previous errors")

    def put_result(self, task, config_obj, events_obj, **kwargs):
        now = kwargs.get('date', datetime.datetime.utcnow().isoformat())
        train_events = list(filter(lambda x: x['phase'] == 'Train', events_obj))
        valid_events = list(filter(lambda x: x['phase'] == 'Valid', events_obj))
        test_events = list(filter(lambda x: x['phase'] == 'Test', events_obj))

        checkpoint_base = kwargs.get('checkpoint_base', None)
        checkpoint_store = kwargs.get('checkpoint_store', None)
        print_fn = kwargs.get('print_fn', self.logger.warning)
        hostname = kwargs.get('hostname', socket.gethostname())
        username = kwargs.get('username', getpass.getuser())
        config_sha1 = hash_config(config_obj)
        label = get_experiment_label(config_obj, task, **kwargs)
        checkpoint = kwargs.get('checkpoint', self.get_checkpoint(checkpoint_base, checkpoint_store, config_sha1,
                                                                  hostname))
        version = kwargs.get('version', __version__)
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
            "version": version,
            "checkpoint": checkpoint
        }
        
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
            return None
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

    @staticmethod
    def _get_metrics_mongo(xs):
        keys = []
        for x in xs:
            keys += x.keys()
        keys = set(keys)
        if 'tick_type' in keys:
            keys.remove("tick_type")
        if 'tick' in keys:
            keys.remove("tick")
        if 'phase' in keys:
            keys.remove("phase")
        return keys
        
    def mongo_to_experiment_set(self, task, all_results, event_type, metrics):
        data = []
        event_types = [event_type] if event_type else list(set(EVENT_TYPES.values()))
        metrics_from_user = set(metrics)
        for result in all_results:  # different experiments
            task = task
            _id = result['_id']
            username = result.get('username', 'root')
            hostname = result.get('hostname', 'localhost')
            label = result.get('label', 'default_label')
            dataset = result['config']['dataset']
            date = result['date']
            sha1 = result['sha1']
            config = result['config']
            version = result.get('version', '0.5.0')  # backward compatibility
            for event_type in event_types:
                if not result.get(event_type, []):
                    continue
                metrics_from_db = self._get_metrics_mongo(result[event_type])
                if not metrics_from_user:
                    metrics = list(metrics_from_db)
                elif metrics_from_user - metrics_from_db:
                    return MongoError(message='Metrics [{}] not found in the [{}] database'.format(','.join(
                        list(metrics_from_user - metrics_from_db)), task))
                else:
                    metrics = list(metrics_from_user)
                # for train_events we can have different metrics than test_events
                for record in result[event_type]:  # train_event epoch 0,
                    for metric in metrics:
                        data.append(MongoResult(
                            metric=metric,
                            value=record[metric],
                            task=task,
                            _id=str(_id),
                            username=username,
                            hostname=hostname,
                            label=label,
                            config=config,
                            dataset=dataset,
                            date=date,
                            sha1=sha1,
                            event_type=event_type,
                            epoch=record['tick'],
                            version=version
                        ))
        if not data:
            return MongoError(message='No results from the query')
        rs = MongoResultSet(data=data)
        return rs.experiments()

    @staticmethod
    def _update_query(q, **kwargs):
        query = q
        if not kwargs:
            return query
        else:
            if "id" in kwargs and kwargs["id"]:
                if type(kwargs["id"]) == list:
                    query.update({"_id": {"$in": [ObjectId(x) for x in kwargs["id"]]}})
                else:
                    query.update({"_id": ObjectId(kwargs["id"])})
                return query
            if "label" in kwargs and kwargs["label"]:
                query.update({"label": kwargs["label"]})
            if "username" in kwargs and kwargs["username"]:
                if type(kwargs["username"]) == list:
                    query.update({"username": {"$in": kwargs["username"]}})
                else:
                    query.update({"username": kwargs["username"]})
            if "dataset" in kwargs:
                query.update({"config.dataset": kwargs["dataset"]})
            if "sha1" in kwargs:
                query.update({"sha1": kwargs["sha1"]})
            return query

    @staticmethod
    def _update_projection(event_type):
        projection = {"_id": 1, "sha1": 1, "label": 1, "username": 1, "config.dataset": 1, "date": 1}
        projection.update({event_type: 1})
        return projection

    def list_results(self, task, prop, value, user, metric, sort, event_type):
        event_type = event_type if event_type is not None else 'test_events'

        metrics = listify(metric)
        users = listify(user)
        d = {'username': users, prop: value}

        coll = self.db[task]
        query = self._update_query({}, **d)
        all_results = list(coll.find(query))

        if not all_results:
            return MongoError(message='no information available for [{}]: [{}] in task database [{}]'
                              .format(prop, value, task))
        experiments = self.mongo_to_experiment_set(task, all_results, event_type=event_type, metrics=metrics)
        if type(experiments) == MongoError:
            return experiments
        if sort is None:
            return experiments
        else:
            if event_type == 'test_events':
                if sort in METRICS_SORT_ASCENDING:
                    return experiments.sort(sort, reverse=False)
                else:
                    return experiments.sort(sort)
            else:
                return MongoError(message='experiments can only be sorted when event_type=test_results')

    def get_experiment_details(self, task, _id):
        coll = self.db[task]
        query = {'_id': ObjectId(_id)}
        all_results = list(coll.find(query))
        if not all_results:
            return MongoError(message='no experiment with id [{}] for task [{}]'.format(_id, task))
        experiments = self.mongo_to_experiment_set(task, all_results, event_type=None, metrics=[])
        if type(experiments) == MongoError:
            return experiments
        return experiments[0]

    @staticmethod
    def aggregate_results(resultset, reduction_dim, event_type, num_exps_per_reduction):
        # TODO: implement a trim method for ExperimentGroup
        grouped_result = resultset.groupby(reduction_dim)
        aggregate_fns = {'min': np.min, 'max': np.max, 'avg': np.mean, 'std': np.std}
        return grouped_result.reduce(aggregate_fns=aggregate_fns, event_type=event_type)

    def get_results(self, task, prop, value, reduction_dim, metric, sort, numexp_reduction_dim, event_type):
        metrics = listify(metric)
        event_type = event_type if event_type is not None else 'test_events'
        reduction_dim = reduction_dim if reduction_dim is not None else 'sha1'
        coll = self.db[task]
        d = {prop: value}
        query = self._update_query({}, **d)
        all_results = list(coll.find(query))
        if not all_results:
            return MongoError(message='no information available for [{}]: [{}] in task database [{}]'
                              .format(prop, value, task))
        resultset = self.mongo_to_experiment_set(task, all_results, event_type=event_type, metrics=metrics)
        if type(resultset) is not MongoError:
            return self.aggregate_results(resultset, reduction_dim, event_type, numexp_reduction_dim)
        return resultset

    def config2json(self, task, sha1):
        coll = self.db[task]
        j = coll.find_one({"sha1": sha1}, {"config": 1})
        if not j:
            return MongoError('no config [{}] in [{}] database'.format(sha1, task))
        else:
            return j["config"]

    def get_task_names(self):
        return self.db.collection_names()

    def get_model_location(self, task, eid):
        coll = self.db[task]
        query = {'_id': ObjectId(eid)}
        projection = {'checkpoint': 1}
        results = [x.get('checkpoint') for x in list(coll.find(query, projection))]
        results = [x for x in results if x is not None]
        if not results:
            return MongoError(message='no model location for experiment id [{}] in [{}] database'.format(eid, task))
        return results[0]

    def task_summary(self, task):
        event_type = 'test_events'
        coll = self.db[task]
        q = {}
        p = {'config.dataset': 1}
        datasets = list(set([x['config']['dataset'] for x in list(coll.find(q, p))]))
        store = []
        for dataset in datasets:
            q = self._update_query({}, dataset=dataset)
            p = self._update_projection(event_type)
            results = list(coll.find(q, p))
            experiment_set = self.mongo_to_experiment_set(task, results, event_type, metrics=[])
            if type(experiment_set) == MongoError:
                self.logger.error('Error getting summary for task [{}], dataset [{}], stacktrace [{}]'
                                  .format(task, dataset, experiment_set.message))
                continue
            store.append(MongoTaskDatasetSummary(task=task, dataset=dataset,
                                                 experiment_set=self.mongo_to_experiment_set(task, results, event_type,
                                                                                             metrics=[])))
        return MongoTaskDatasetSummarySet(task=task, data=store).groupby()
    
    def summary(self):
        tasks = self.get_task_names()
        if "system.indexes" in tasks:
            tasks.remove("system.indexes")
        return [self.task_summary(task) for task in tasks]
        
    def dump(self, zip='xpctl-mongodump-{}'.format(datetime.datetime.now().isoformat()), task_eids={}):
        """ dump reporting log and config for later consumption"""
        events = ['train_events', 'valid_events', 'test_events']
        tasks = self.get_task_names() if not task_eids.keys() else list(task_eids.keys())
        if "system.indexes" in tasks:
            tasks.remove("system.indexes")

        base_dir = '/tmp/xpctldump'
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
            
        os.makedirs(base_dir, exist_ok=True)
        
        for task in tasks:
            coll = self.db[task]
            query = self._update_query({}, id=listify(task_eids.get(task, [])))
            all_results = list(coll.find(query))
            for result in all_results:
                _id = str(result['_id'])
                result.pop('_id')
                _dir = os.path.join(base_dir, task, _id)
                os.makedirs(_dir, exist_ok=True)
                config = result['config']
                result.pop('config')
                write_config_file(config, os.path.join(_dir, '{}-config.yml'.format(_id)))
                with open(os.path.join(_dir, '{}-reporting.log'.format(_id)), 'w') as f:
                    for event in events:
                        for item in result.get(event, []):
                            f.write(json.dumps(item)+'\n')
                        result.pop(event)
                write_config_file(result, os.path.join(_dir, '{}-meta.yml'.format(_id)))
                
        return shutil.make_archive(base_name=zip, format='zip', root_dir='/tmp', base_dir='xpctldump')
    
    def restore(self, dump):
        """ if dump is in zip format, will unzip it. expects the following dir structure in the unzipped file:
        <root>
         - <task>
           - <id>-reporting.log
           - <id>.yml
        """
        dump_dir = unzip_files(dump)
        for task in os.listdir(dump_dir):
            task_dir = os.path.join(dump_dir, task)
            for exp in os.listdir(task_dir):
                exp_dir = os.path.join(task_dir, exp)
                meta = [os.path.join(exp_dir, x) for x in os.listdir(exp_dir) if x.endswith('meta.yml')]
                reporting = [os.path.join(exp_dir, x) for x in os.listdir(exp_dir) if x.endswith('reporting.log')]
                config = [os.path.join(exp_dir, x) for x in os.listdir(exp_dir) if x.endswith('config.yml')]
                try:
                    assert len(config) == 1
                    assert len(reporting) == 1
                    assert len(meta) == 1
                    config = read_config_file(config[0])
                    meta = read_config_file(meta[0])
                    reporting = log2json(reporting[0])
                except AssertionError:
                    raise RuntimeError('There should be exactly one meta file, one config file and one reporting log '
                                       'in {}'.format(exp_dir))
                self.put_result(task, config_obj=config, events_obj=reporting, **meta)
        if dump_dir != dump:
            shutil.rmtree(dump_dir)

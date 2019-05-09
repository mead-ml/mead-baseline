import os
import shutil
import pymongo
import datetime
import socket
import getpass
from baseline.utils import export, listify, read_config_file, write_config_file, unzip_files
from mead.utils import hash_config
from xpctl.backends.core import ExperimentRepo
from xpctl.backends.helpers import store_model
from xpctl.backends.mongo.dto import mongo_to_experiment_set
from xpctl.backends.dto import experiment_to_put_result_consumable, aggregate_results, write_experiment
from backends.data import TaskDatasetSummary, TaskDatasetSummarySet, Success, Error
from xpctl.backends.helpers import log2json, get_experiment_label, METRICS_SORT_ASCENDING, get_checkpoint
from bson.objectid import ObjectId
from baseline.version import __version__
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

    def put_result(self, task, exp):
        unpacked = experiment_to_put_result_consumable(exp)
        return self._put_result(task=task, config_obj=unpacked.config_obj, events_obj=unpacked.events_obj,
                                **unpacked.extra_args)
        
    def _put_result(self, task, config_obj, events_obj, **kwargs):
        now = kwargs.get('date', datetime.datetime.utcnow().isoformat())
        hostname = kwargs.get('hostname', socket.gethostname())
        username = kwargs.get('username', getpass.getuser())
        config_sha1 = kwargs.get('sha1', hash_config(config_obj))
        label = kwargs.get('label', get_experiment_label(config_obj, task, **kwargs))
        checkpoint_base = kwargs.get('checkpoint_base')
        checkpoint_store = kwargs.get('checkpoint_store')
        checkpoint = kwargs.get('checkpoint', get_checkpoint(checkpoint_base, checkpoint_store, config_sha1,
                                                              hostname))
        version = kwargs.get('version', __version__)

        train_events = list(filter(lambda x: x['phase'] == 'Train', events_obj))
        valid_events = list(filter(lambda x: x['phase'] == 'Valid', events_obj))
        test_events = list(filter(lambda x: x['phase'] == 'Test', events_obj))

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
        
        if '_id' in kwargs:
            post.update({'_id': ObjectId(kwargs['_id'])})
            
        try:
            coll = self.db[task]
            result = coll.insert_one(post)
            return Success(message='experiment successfully inserted: {}'.format(result.inserted_id))
        except pymongo.errors.PyMongoError as e:
            return Error(message='experiment could not be inserted: {}'.format(e.message))

    def put_model(self, eid, task, checkpoint_base, checkpoint_store, print_fn=print):
        coll = self.db[task]
        query = {'_id': ObjectId(eid)}
        projection = {'sha1': 1}
        results = list(coll.find(query, projection))
        if not results:
            print_fn("no sha1 for the given id found, returning.")
            return None
        sha1 = results[0]['sha1']
        model_loc = store_model(checkpoint_base, sha1, checkpoint_store, print_fn)
        if model_loc is not None:
            coll.update_one({'_id': ObjectId(eid)}, {'$set': {'checkpoint': model_loc}}, upsert=False)
        return model_loc

    def get_model_location(self, task, eid):
        coll = self.db[task]
        query = {'_id': ObjectId(eid)}
        projection = {'checkpoint': 1}
        results = [x.get('checkpoint') for x in list(coll.find(query, projection))]
        results = [x for x in results if x is not None]
        if not results:
            return Error(message='no model location for experiment id [{}] in [{}] database'.format(eid, task))
        return results[0]

    def update_prop(self, task, eid, prop, value):
        try:
            coll = self.db[task]
            r = coll.find_one({'_id': ObjectId(eid)}, {prop: 1})
            if r is None:
                return Error(message='property {} for experiment {} not found in {} database'.format(prop, eid, task))
            prev_value = r[prop]
            coll.update({'_id': ObjectId(eid)}, {'$set': {prop: value}}, upsert=False)
            changed_value = coll.find_one({'_id': ObjectId(eid)}, {prop: 1})[prop]
            return Success(message='for experiment [{}] property [{}] was changed from [{}] to [{}]'
                           .format(eid, prop, prev_value, changed_value))
        except pymongo.errors.PyMongoError as e:
            return Error(message='property update failed: {}'.format(e.message))

    def remove_experiment(self, task, eid):
        try:
            coll = self.db[task]
            prev = coll.find_one({'_id': ObjectId(eid)})
            if prev is None:
                return Error(message='delete operation failed: experiment [{}] not found in [{}] database'.format(eid, task))
            model_loc = self.get_model_location(task, eid)
            if model_loc is not None and type(model_loc) is not Error and os.path.exists(model_loc):
                try:
                    os.remove(model_loc)
                except IOError:
                    return Error(message='model {} exists on host but could not be removed'.format(model_loc))
            coll.remove({'_id': ObjectId(eid)})
            try:
                assert coll.find_one({'_id': ObjectId(eid)}) is None
                return Success("record [{}] deleted successfully from database [{}]".format(eid, task))
            except AssertionError:
                return Error('delete failed: could not delete experiment {} from {} database'.format(eid, task))
        except pymongo.errors.PyMongoError as e:
            return Error(message='experiment could not be removed: {}'.format(e.message))

    def get_experiment_details(self, task, eid, event_type, metric):
        metrics = [x for x in listify(metric) if x.strip()]
        event_type = event_type if event_type is not None else 'test_events'

        coll = self.db[task]
        query = {'_id': ObjectId(eid)}
        all_results = list(coll.find(query))
        if not all_results:
            return Error(message='no experiment with id [{}] for task [{}]'.format(eid, task))
        experiments = mongo_to_experiment_set(task, all_results, event_type=event_type, metrics=metrics)
        if type(experiments) == Error:
            return Error(experiments.message)
        return experiments[0]

    def get_results(self, task, prop, value, reduction_dim, metric, sort, numexp_reduction_dim, event_type):
        metrics = [x for x in listify(metric)]
        event_type = event_type if event_type is not None else 'test_events'
        reduction_dim = reduction_dim if reduction_dim is not None else 'sha1'
        coll = self.db[task]
        d = {prop: value}
        query = self._update_query({}, **d)
        all_results = list(coll.find(query))
        if not all_results:
            return Error(message='no information available for [{}]: [{}] in task database [{}]'
                         .format(prop, value, task))
        resultset = mongo_to_experiment_set(task, all_results, event_type=event_type, metrics=metrics)
        if type(resultset) is not Error:
            return aggregate_results(resultset, reduction_dim, event_type, numexp_reduction_dim)
        return resultset

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

        metrics = [x for x in listify(metric) if x.strip()]
        users = [x for x in listify(user) if x.strip()]
        d = {prop: value}
        if users:
            d.update({'username': users})
        coll = self.db[task]
        query = self._update_query({}, **d)
        all_results = list(coll.find(query))

        if not all_results:
            return Error(message='no information available for [{}]: [{}] in task database [{}]'
                         .format(prop, value, task))
        experiments = mongo_to_experiment_set(task, all_results, event_type=event_type, metrics=metrics)
        if type(experiments) == Error:
            return experiments
        if sort is None or (type(sort) == str and sort == 'None'):
            return experiments
        else:
            if event_type == 'test_events':
                if sort in METRICS_SORT_ASCENDING:
                    return experiments.sort(sort, reverse=False)
                else:
                    return experiments.sort(sort)
            else:
                return Error(message='experiments can only be sorted when event_type=test_results')

    def config2json(self, task, sha1):
        coll = self.db[task]
        j = coll.find_one({"sha1": sha1}, {"config": 1})
        if not j:
            return Error('no config [{}] in [{}] database'.format(sha1, task))
        else:
            return j["config"]

    def get_task_names(self):
        return self.db.collection_names()

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
            experiment_set = mongo_to_experiment_set(task, results, event_type, metrics=[])
            if type(experiment_set) == Error:
                self.logger.error('Error getting summary for task [{}], dataset [{}], stacktrace [{}]'
                                  .format(task, dataset, experiment_set.message))
                continue
            store.append(TaskDatasetSummary(task=task, dataset=dataset, experiment_set=experiment_set))
        if not store:
            return Error('could not get summary for task: [{}]'.format(task))
        return TaskDatasetSummarySet(task=task, data=store).groupby()
    
    def summary(self):
        tasks = self.get_task_names()
        if "system.indexes" in tasks:
            tasks.remove("system.indexes")
        return [self.task_summary(task) for task in tasks]
        
    def dump(self, zipfile='xpctldump-{}'.format(datetime.datetime.now().isoformat()), task_eids={}):
        """ dump reporting log and config for later consumption"""
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
            experiments = mongo_to_experiment_set(task, all_results, event_type=[], metrics=[]).data
            _dir = os.path.join(base_dir, task)
            os.makedirs(_dir)
            for exp in experiments:
                write_experiment(exp, _dir)
        return shutil.make_archive(base_name=zipfile, format='zip', root_dir='/tmp', base_dir='xpctldump')
    
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
                self._put_result(task, config_obj=config, events_obj=reporting, **meta)
        if dump_dir != dump:
            shutil.rmtree(dump_dir)

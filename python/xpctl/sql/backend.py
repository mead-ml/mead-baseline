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

from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
import sqlalchemy as sql
import sqlalchemy.orm as orm
from xpctl.core import ExperimentRepo, store_model


EVENT_TYPES = {
    "train": "train_events", "Train": "train_events",
    "test": "test_events", "Test": "test_events",
    "valid": "valid_events", "Valid": "valid_events",
    "dev": "valid_events", "Dev": "valid_events"
}


@exporter
class Metric(Base):

    __tablename__ = 'metrics'

    id = sql.Column(sql.Integer, primary_key=True)
    label = sql.Column(sql.String, nullable=False)
    value = sql.Column(sql.Float, nullable=False)
    event_id = sql.Column(sql.Integer, sql.ForeignKey('events.id'))
    event = orm.relationship('Event', back_populates='metrics')


@exporter
class Experiment(Base):

    __tablename__ = 'experiments'
    id = sql.Column(sql.Integer, primary_key=True)
    label = sql.Column(sql.String, nullable=False)
    sha1 = sql.Column(sql.String, nullable=False)
    task = sql.Column(sql.String, nullable=False)
    dataset = sql.Column(sql.String, nullable=False)
    config = sql.Column(sql.String, nullable=False)
    hostname = sql.Column(sql.String, nullable=False)
    username = sql.Column(sql.String, nullable=False)
    version = sql.Column(sql.String)
    date = sql.Column(sql.DateTime, nullable=True)
    status = sql.Column(sql.String, nullable=False)
    last_modified = sql.Column(sql.DateTime, nullable=False)
    checkpoint = sql.Column(sql.String)
    events = orm.relationship('Event', back_populates='experiment')


@exporter
class Event(Base):
    __tablename__ = 'events'
    id = sql.Column(sql.Integer, primary_key=True)
    phase = sql.Column(sql.String, nullable=False)
    experiment_id = sql.Column(sql.Integer, sql.ForeignKey('experiments.id'), nullable=False)
    experiment = orm.relationship('Experiment', back_populates="events")
    tick = sql.Column(sql.Integer, nullable=False)
    date = sql.Column(sql.DateTime)
    metrics = orm.relationship('Metric', back_populates='event')


@exporter
class SQLRepo(ExperimentRepo):

    def _connect(self, uri):
        self.engine = sql.create_engine(uri, echo=True, paramstyle='format')
        Base.metadata.create_all(self.engine)
        self.Session = orm.sessionmaker(bind=self.engine)

    def __init__(self, **kwargs):
        super(SQLRepo, self).__init__()

        uri = kwargs.get('uri', None)
        if uri is None:
            dbtype = kwargs.get('dbtype', 'postgresql')
            host = kwargs.get('host', 'localhost')
            port = kwargs.get('port', None)
            if port is not None:
                host = '{}:{}'.format(host, port)
            username = kwargs.get('user', None)
            passwd = kwargs.get('passwd', None)

            user = username if passwd is None else '{}:{}'.format(username, passwd)
            dbname = kwargs.get('db', 'reporting_db')
            uri = '{}://{}@{}/{}'.format(dbtype, user, host, dbname)
        self._connect(uri)

    def _find_or_create(self, model, **kwargs):
        session = self.Session()
        instance = session.query(model).filter_by(**kwargs).first()
        if instance:
            return instance, False
        else:
            instance = model(**kwargs)
            session.add(instance)
            session.commit()
            return instance, True

    def put_result(self, task, config_obj, events_obj, **kwargs):
        session = self.Session()
        print_fn = kwargs.get('print_fn', print)
        now = datetime.datetime.utcnow().isoformat()
        hostname = kwargs.get('hostname', socket.gethostname())
        username = kwargs.get('username', getpass.getuser())
        config_sha1 = hashlib.sha1(json.dumps(config_obj).encode('utf-8')).hexdigest()
        label = kwargs.get("label", config_sha1)
        checkpoint_base = kwargs.get('checkpoint_base', None)
        checkpoint_store = kwargs.get('checkpoint_store', None)

        checkpoint = None
        if checkpoint_base:
            model_loc = store_model(checkpoint_base, config_sha1, checkpoint_store)
            if model_loc is not None:
                checkpoint = "{}:{}".format(hostname, os.path.abspath(model_loc))
            else:
                print_fn("model could not be stored, see previous errors")

        event_objs = []
        for event in events_obj:
            tick = event['tick']
            date = event.get('date')
            phase = event['phase']
            event_obj = Event(tick=tick, date=date, phase=phase, metrics=[])
            for key in event.keys():
                if key not in ['tick_type', 'tick', 'event_type', 'id', 'date', 'phase']:
                    metric = Metric(label=key, value=event[key])
                    event_obj.metrics += [metric]
            event_objs += [event_obj]

        experiment = Experiment(
            label=label,
            checkpoint=checkpoint,
            sha1=config_sha1,
            task=task,
            dataset=config_obj['dataset'],
            config=json.dumps(config_obj),
            hostname=hostname,
            username=username,
            date=now,
            version=__version__,
            status='CREATED',
            last_modified=now
        )
        experiment.events = event_objs
        session.add(experiment)
        session.commit()
        return experiment

    def has_task(self, task):
        return task in self.get_task_names()

    def get_task_names(self):
        session = self.Session()
        return set([t[0] for t in session.query(Experiment.task).distinct()])

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
        session = self.Session()
        exp = session.query(Experiment).get(id)
        return exp.label

    def rename_label(self, id, task, new_label):
        session = self.Session()
        exp = session.query(Experiment).get(id)
        old_label = exp.label
        exp.label = new_label
        session.commit()
        return old_label, new_label

    def rm(self, id, task, print_fn=print):
        session = self.Session()
        exp = session.query(Experiment).get(id)
        exp.delete(synchronize_session=False)
        return True

    def event2phase(self, event_type):
        if event_type == 'train_events':
            return 'Train'
        if event_type == 'valid_events':
            return 'Valid'
        if event_type == 'test_events':
            return 'Test'
        raise Exception('Unknown event type {}'.event_type)

    def _nbest_by_metric_rows(self, username, metric, dataset, task, num_results, event_type, ascending):
        session = self.Session()
        phase = self.event2phase(event_type)

        best = session.query(Experiment, Metric, Event).\
            filter(Experiment.dataset == dataset). \
            filter(Event.phase == phase).\
            filter(Experiment.task == task).\
            filter(Event.experiment_id == Experiment.id).\
            filter(Event.id == Metric.event_id).\
            filter(Metric.label == metric)

        if username is not None and len(username) > 0:
            best = best.filter(Experiment.username.in_(username))

        best = best.order_by(Metric.value.asc() if ascending is True else Metric.value.desc())
        if num_results > 0:
            return best[0:num_results]
        return best

    def nbest_by_metric(self, username, metric, dataset, task, num_results, event_type, ascending):
        best = self._nbest_by_metric_rows(username, metric, dataset, task, num_results, event_type, ascending)
        results = []
        for exp, metric, event in best:
            results.append([exp.id, exp.username, exp.label, exp.dataset, exp.sha1, exp.date, metric.value])
        return pd.DataFrame(results, columns=['id', 'username', 'label', 'dataset', 'sha1', 'date', metric.label])

    def get_results(self, username, metric, sort, dataset, task, event_type):
        session = self.Session()

        results = []
        metrics = listify(metric)
        metrics_to_add = [metrics[0]] if len(metrics) == 1 else []
        phase = self.event2phase(event_type)
        if len(metric) == 1:
            metric = metric[0]
            if metric == "avg_loss" or metric == "perplexity":
                ascending = True
            else:
                ascending = False

            if sort:
                if sort == "avg_loss" or sort == "perplexity":
                    ascending = True
                else:
                    ascending = False
            best = self._nbest_by_metric_rows(username, metric, dataset, task, -1, event_type, ascending)
            for exp, metric, _ in best:
                result = [exp.id, exp.username, exp.label, exp.dataset, exp.sha1, exp.date, metric.value]
                for event in exp.events:
                    if phase == event.phase:
                        for m in event.metrics:
                            if m.label != metric.label:
                                result += [m.value]
                                if m.label not in metrics_to_add:
                                    metrics_to_add += [m.label]
                results.append(result)
            return pd.DataFrame(results,
                                columns=['id', 'username', 'label', 'dataset', 'sha1', 'date'] + metrics_to_add)

        phase = self.event2phase(event_type)

        hits = session.query(Experiment).filter(Experiment.dataset == dataset). \
            filter(Experiment.task == task)

        for exp in hits:
            for event in exp.events:

                if event.phase == phase:
                    result = [exp.id, exp.username, exp.label, exp.dataset, exp.sha1, exp.date]
                    for m in event.metrics:
                        result += [m.value]
                        if m.label not in metrics_to_add:
                            metrics_to_add += [m.label]
                    results.append(result)
        cols = ['id', 'username', 'label', 'dataset', 'sha1', 'date'] + metrics_to_add
        return pd.DataFrame(results, columns=cols)

    def task_summary(self, task, dataset, metric, event_type):

        ascending = True if metric == "avg_loss" or metric == "perplexity" else False
        exp, metric, _ = self._nbest_by_metric_rows(None, metric, dataset, task, 1, event_type, ascending)[0]
        summary = "For dataset {}, the best {} is {:0.3f} reported by {} on {}. " \
                      "The sha1 for the config file is {}.".format(exp.dataset,
                                                                   metric.label,
                                                                   metric.value,
                                                                   exp.username,
                                                                   exp.date,
                                                                   exp.sha1)
        return summary

    def phase2event(self, phase):
        return EVENT_TYPES[phase]

    def config2dict(self, task, sha1):
        session = self.Session()
        config = session.query(Experiment).filter(Experiment.sha1 == sha1).one().config
        return json.loads(config)

    def get_info(self, task, event_types):
        session = self.Session()
        if task is not None:
            results = []
            rs = session.query(Experiment).filter(Experiment.task == task)
            for exp in rs:
                for event in exp.events:
                    event_type = self.phase2event(event.phase)
                    if event_types is None or event_type in event_types:
                        result = [exp.username, exp.dataset, event_type] + [','.join([m.label for m in event.metrics])]
                    results.append(result)

        df = pd.DataFrame(results, columns=['user', 'dataset', 'event_type', 'metrics'])
        return df.groupby(['user', 'dataset', 'event_type', 'metrics']).size().reset_index() \
            .rename(columns={0: 'num_experiments'})
        return results

    def leaderboard_summary(self, task=None, event_types=None, print_fn=print):

        if task:
            print_fn("Task: [{}]".format(task))
            print_fn("-" * 93)
            print_fn(self.get_info(task, event_types))
        else:
            tasks = self.get_task_names()
            print_fn("There are {} tasks: {}".format(len(tasks), tasks))
            for task in tasks:
                print_fn("-" * 93)
                print_fn("Task: [{}]".format(task))
                print_fn("-" * 93)
                print_fn(self.get_info(task, event_types))


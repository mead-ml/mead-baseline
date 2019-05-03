from __future__ import print_function
import os
import pandas as pd
import datetime
import socket
import json
import getpass
from baseline.utils import export, listify
from mead.utils import hash_config
from baseline.version import __version__
from xpctl.helpers import df_get_results, df_experimental_details

__all__ = []
exporter = export(__all__)

from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
import sqlalchemy as sql
import sqlalchemy.orm as orm
from xpctl.backend.core import ExperimentRepo, store_model
from xpctl.helpers import get_experiment_label

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
        self.engine = sql.create_engine(uri, echo=False, paramstyle='format')
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

    def put_result(self, task, config_obj, events_obj, **kwargs):
        session = self.Session()
        print_fn = kwargs.get('print_fn', print)
        now = datetime.datetime.utcnow().isoformat()
        hostname = kwargs.get('hostname', socket.gethostname())
        username = kwargs.get('username', getpass.getuser())
        config_sha1 = hash_config(config_obj)
        label = get_experiment_label(config_obj, task, **kwargs)
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
        session = self.Session()
        exp = session.query(Experiment).get(id)
        if exp is None:
            print_fn("no sha1 for the given id found, returning.")
            return None
        sha1 = exp.sha1
        model_loc = store_model(checkpoint_base, sha1, checkpoint_store, print_fn)
        if model_loc is not None:
            exp.checkpoint = model_loc
            session.commit()
        return model_loc

    def get_model_location(self, id, task):
        session = self.Session()
        exp = session.query(Experiment).get(id)
        if exp is None:
            return None
        return exp.checkpoint

    def get_label(self, id, task):
        session = self.Session()
        exp = session.query(Experiment).get(id)
        if exp is None:
            return None
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

    @staticmethod
    def _get_filtered_metrics(allmetrics, metrics):
        if not metrics:
            return allmetrics
        else:
            return [x for x in allmetrics if x.label in metrics]

    def get_results(self, task, dataset, event_type, num_exps=None, num_exps_per_config=None, metric=None, sort=None, id=None, label=None):
        session = self.Session()
        results = []
        metrics = listify(metric)
        metrics_to_add = [metrics[0]] if len(metrics) == 1 else []
        phase = self.event2phase(event_type)
        if id is not None:
            hit = session.query(Experiment).get(id)
            if hit is None:
                return None
            hits = [hit]
        elif label is not None:
            hits = session.query(Experiment).filter(Experiment.label == label). \
                filter(Experiment.dataset == dataset). \
                filter(Experiment.task == task)
        else:
            hits = session.query(Experiment).filter(Experiment.dataset == dataset). \
                filter(Experiment.task == task)
        for exp in hits:
            for event in exp.events:
                if event.phase == phase:
                    result = [exp.id, exp.username, exp.label, exp.dataset, exp.sha1, exp.date]
                    for m in self._get_filtered_metrics(event.metrics, metrics):
                        result += [m.value]
                        if m.label not in metrics_to_add:
                            metrics_to_add += [m.label]
                    results.append(result)
        cols = ['id', 'username', 'label', 'dataset', 'sha1', 'date'] + metrics_to_add
        result_frame = pd.DataFrame(results, columns=cols)
        if not result_frame.empty:
            return df_get_results(result_frame, dataset, num_exps, num_exps_per_config, metric, sort)
        return None

    def experiment_details(self, user, metric, sort, task, event_type, sha1, n):
        session = self.Session()
        results = []
        metrics = listify(metric)
        users = listify(user)
        metrics_to_add = [metrics[0]] if len(metrics) == 1 else []
        phase = self.event2phase(event_type)
        hits = session.query(Experiment).filter(Experiment.sha1 == sha1). \
            filter(Experiment.task == task)
        for exp in hits:
            for event in exp.events:
                if event.phase == phase:
                    result = [exp.id, exp.username, exp.label, exp.dataset, exp.sha1, exp.date]
                    for m in self._get_filtered_metrics(event.metrics, metrics):
                        result += [m.value]
                        if m.label not in metrics_to_add:
                            metrics_to_add += [m.label]
                    results.append(result)
        cols = ['id', 'username', 'label', 'dataset', 'sha1', 'date'] + metrics_to_add
        result_frame = pd.DataFrame(results, columns=cols)
        return df_experimental_details(result_frame, sha1, users, sort, metric, n)

    def phase2event(self, phase):
        return EVENT_TYPES[phase]

    def config2dict(self, task, sha1):
        session = self.Session()
        config = session.query(Experiment).filter(Experiment.sha1 == sha1).one().config
        return json.loads(config)

    def get_info(self, task, event_type):
        session = self.Session()
        results = []
        rs = session.query(Experiment).filter(Experiment.task == task)
        for exp in rs:
            for event in exp.events:
                current_event_type = self.phase2event(event.phase)
                if event_type is None or current_event_type == event_type:
                    result = [exp.username, exp.dataset, task]
                    results.append(result)
        df = pd.DataFrame(results, columns=['user', 'dataset', 'task'])
        return df.groupby(['user', 'dataset']).agg([len]) \
            .rename(columns={"len": 'num_exps'})

    def leaderboard_summary(self, task=None, event_type=None, print_fn=print):
        if task:
            print_fn("Task: [{}]".format(task))
            print_fn("-" * 93)
            print_fn(self.get_info(task, event_type))
        else:
            tasks = self.get_task_names()
            if "system.indexes" in tasks:
                tasks.remove("system.indexes")
            print_fn("There are {} tasks: {}".format(len(tasks), tasks))
            for task in tasks:
                print_fn("-" * 93)
                print_fn("Task: [{}]".format(task))
                print_fn("-" * 93)
                print_fn(self.get_info(task, event_type))


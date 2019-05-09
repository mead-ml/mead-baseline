from __future__ import print_function
import os
import datetime
import socket
import json
import getpass
from baseline.utils import export, listify
from mead.utils import hash_config
from baseline.version import __version__
from xpctl.backend.helpers import log2json, get_experiment_label, METRICS_SORT_ASCENDING, get_checkpoint, store_model
from xpctl.backend.dto import unpack_experiment
from xpctl.backend.data import Error, Success, TaskDatasetSummary, TaskDatasetSummarySet
from xpctl.backend.sql.dto import sql_result_to_data_experiment, aggregate_sql_results, get_data_experiment_set
from baseline.utils import unzip_files, read_config_file
import shutil

__all__ = []
exporter = export(__all__)

from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
import sqlalchemy as sql
import sqlalchemy.orm as orm
from xpctl.backend.core import ExperimentRepo

EVENT_TYPES = {
    "train": "train_events", "Train": "train_events",
    "test": "test_events", "Test": "test_events",
    "valid": "valid_events", "Valid": "valid_events",
    "dev": "valid_events", "Dev": "valid_events"
}


@exporter
class SqlResult(Base):
    __tablename__ = 'results'
    id = sql.Column(sql.Integer, primary_key=True)
    metric = sql.Column(sql.String, nullable=False)
    value = sql.Column(sql.Float, nullable=False)
    event_id = sql.Column(sql.Integer, sql.ForeignKey('events.id'))
    event = orm.relationship('SqlEvent', back_populates='results', cascade='all,delete')


@exporter
class SqlEvent(Base):
    __tablename__ = 'events'
    id = sql.Column(sql.Integer, primary_key=True)
    phase = sql.Column(sql.String, nullable=False)
    tick_type = sql.Column(sql.String, nullable=False)
    tick = sql.Column(sql.Integer, nullable=False)
    experiment_id = sql.Column(sql.Integer, sql.ForeignKey('experiments.eid'), nullable=False)
    experiment = orm.relationship('SqlExperiment', back_populates='events', cascade='all,delete')
    results = orm.relationship('SqlResult', back_populates='event', cascade='all,delete')


@exporter
class SqlExperiment(Base):
    __tablename__ = 'experiments'
    eid = sql.Column(sql.Integer, primary_key=True)
    label = sql.Column(sql.String)
    sha1 = sql.Column(sql.String)
    task = sql.Column(sql.String)
    dataset = sql.Column(sql.String)
    config = sql.Column(sql.String)
    hostname = sql.Column(sql.String)
    username = sql.Column(sql.String)
    version = sql.Column(sql.String)
    date = sql.Column(sql.DateTime)
    status = sql.Column(sql.String)
    last_modified = sql.Column(sql.DateTime)
    checkpoint = sql.Column(sql.String)
    events = orm.relationship('SqlEvent', back_populates='experiment', cascade='all,delete')


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

    def put_result(self, task, exp):
        unpacked = unpack_experiment(exp)
        return self._put_result(task=task, config_obj=unpacked.config_obj, events_obj=unpacked.events_obj,
                                **unpacked.extra_args)
    
    def _put_result(self, task, config_obj, events_obj, **kwargs):
        session = self.Session()
        
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

        events = []
        for event in events_obj:
            tick = event['tick']
            phase = event['phase']
            event_obj = SqlEvent(phase=phase, tick_type=event['tick_type'], tick=tick, results=[])
            for key in event.keys():
                if key not in ['tick_type', 'tick', 'event_type', 'id', 'date', 'phase']:
                    metric = SqlResult(metric=key, value=event[key])
                    event_obj.results += [metric]
            events += [event_obj]
        experiment = SqlExperiment(
            label=label,
            checkpoint=checkpoint,
            sha1=config_sha1,
            task=task,
            dataset=config_obj['dataset'],
            config=json.dumps(config_obj),
            hostname=hostname,
            username=username,
            date=now,
            version=version,
            status='CREATED',
            last_modified=now,
            events=events
        )
        try:
            session.add(experiment)
            session.commit()
            return Success(message='experiment successfully inserted: {}'.format(experiment.eid))
        except sql.exc.SQLAlchemyError as e:
            return Error(message=str(e))
        
    def put_model(self, id, task, checkpoint_base, checkpoint_store, print_fn=print):
        session = self.Session()
        exp = session.query(SqlExperiment).get(id)
        if exp is None:
            print_fn("no sha1 for the given id found, returning.")
            return None
        sha1 = exp.sha1
        model_loc = store_model(checkpoint_base, sha1, checkpoint_store, print_fn)
        if model_loc is not None:
            exp.checkpoint = model_loc
            session.commit()
        return model_loc

    def get_model_location(self, task, eid):
        session = self.Session()
        exp = session.query(SqlExperiment).get(eid)
        if exp is None:
            return Error(message='no model location for experiment id [{}] in [{}] database'.format(eid, task))
        return exp.checkpoint

    def update_prop(self, task, eid, prop, value):
        try:
            session = self.Session()
            # there must be a better way of getting a column value through a column name
            prev_value = getattr(session.query(SqlExperiment).get(eid), prop)
            session.query(SqlExperiment).filter(SqlExperiment.eid == eid).update({prop: value})
            session.commit()
            changed_value = getattr(session.query(SqlExperiment).get(eid), prop)
            return Success(message='for experiment [{}] property [{}] was changed from [{}] to [{}].'
                           .format(eid, prop, prev_value, changed_value))
        except sql.exc.SQLAlchemyError as e:
            return Error(message=str(e))

    def remove_experiment(self, task, eid):
        try:
            session = self.Session()
            exp = session.query(SqlExperiment).get(eid)
            if exp is None:
                return Error(message='delete failed: experiment {} not found in {} database'.format(eid, task))
            model_loc = self.get_model_location(task, eid)
            if model_loc is not None and type(model_loc) is not Error and os.path.exists(model_loc):
                try:
                    os.remove(model_loc)
                except IOError:
                    return Error(message='model {} exists on host but could not be removed'.format(model_loc))
            session.delete(exp)
            session.commit()
            try:
                assert session.query(SqlExperiment).get(eid) is None
                return Success("record [{}] deleted successfully from database [{}]".format(eid, task))
            except AssertionError:
                return Error('delete failed: could not delete experiment {} from {} database'.format(eid, task))
        except sql.exc.SQLAlchemyError as e:
            return Error(message=str(e))

    def get_experiment_details(self, task, eid, event_type, metric):
        metrics = [x for x in listify(metric) if x.strip()]
        event_type = event_type if event_type is not None else 'test_events'
        session = self.Session()
        exp = session.query(SqlExperiment).get(eid)
        if exp is None:
            return Error(message='no experiment with id [{}] for task [{}]'.format(eid, task))
        return sql_result_to_data_experiment(exp, event_type, metrics)
    
    def get_results(self, task, prop, value, reduction_dim, metric, sort, numexp_reduction_dim, event_type):
        session = self.Session()
        metrics = [x for x in listify(metric)]
        event_type = event_type if event_type is not None else 'test_events'
        reduction_dim = reduction_dim if reduction_dim is not None else 'sha1'
        hits = session.query(SqlExperiment).filter(getattr(SqlExperiment, prop) == value)\
            .filter(SqlExperiment.task == task)
        if hits is None:
            return Error(message='no information available for [{}]: [{}] in task database [{}]'
                         .format(prop, value, task))
        sql_experiments = []
        for exp in hits:
            sql_experiments.append(sql_result_to_data_experiment(exp, event_type, metrics))
        return aggregate_sql_results(sql_experiments, reduction_dim, event_type, numexp_reduction_dim)
 
    def list_results(self, task, prop, value, user, metric, sort, event_type):
        session = self.Session()
        sql_experiments = []
        metrics = [x for x in listify(metric) if x.strip()]
        users = [x for x in listify(user) if x.strip()]
        hits = session.query(SqlExperiment).filter(getattr(SqlExperiment, prop) == value). \
            filter(SqlExperiment.task == task)
        if users:
            hits = hits.filter(SqlExperiment.username.in_(users)).all()
        for exp in hits:
            sql_experiments.append(sql_result_to_data_experiment(exp, event_type, metrics))
        experiment_set = get_data_experiment_set(sql_experiments)
        if sort is None or (type(sort) == str and sort == 'None'):
            return experiment_set
        else:
            if event_type == 'test_events':
                if sort in METRICS_SORT_ASCENDING:
                    return experiment_set.sort(sort, reverse=False)
                else:
                    return experiment_set.sort(sort)
            else:
                return Error(message='experiments can only be sorted when event_type=test_results')
            
    def config2json(self, task, sha1):
        try:
            session = self.Session()
            config = session.query(SqlExperiment).filter(SqlExperiment.sha1 == sha1).one().config
            return json.loads(config)
        except sql.exc.SQLAlchemyError as e:
            return Error(message=str(e))

    def get_task_names(self):
        session = self.Session()
        return set([t[0] for t in session.query(SqlExperiment.task).distinct()])

    def task_summary(self, task):
        session = self.Session()
        datasets = [x[0] for x in session.query(SqlExperiment.dataset).distinct()]
        users = [x[0] for x in session.query(SqlExperiment.username).distinct()]
        store = []
        for dataset in datasets:
            user_num_exps = {}
            for user in users:
                result = len([x.username for x in session.query(SqlExperiment)
                             .filter(SqlExperiment.task == task)
                             .filter(SqlExperiment.dataset == dataset)
                             .filter(SqlExperiment.username == user).distinct()])
                if result != 0:
                    user_num_exps[user] = result
            store.append(TaskDatasetSummary(task=task, dataset=dataset, experiment_set=None, user_num_exps=user_num_exps))
        if not store:
            return Error('could not get summary for task: [{}]'.format(task))
        return TaskDatasetSummarySet(task=task, data=store).groupby()

    def summary(self):
        tasks = self.get_task_names()
        if "system.indexes" in tasks:
            tasks.remove("system.indexes")
        return [self.task_summary(task) for task in tasks]
       
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

from __future__ import print_function
import os
import datetime
import socket
import json
import getpass
from baseline.utils import export, listify
from mead.utils import hash_config
from baseline.version import __version__
from xpctl.backends.backend import log2json, get_experiment_label, METRICS_SORT_ASCENDING, safe_get, \
    client_experiment_to_put_result_consumable, write_experiment, aggregate_results
from xpctl.backends.backend import BackendError, BackendSuccess, TaskDatasetSummary, TaskDatasetSummarySet, Experiment, Result, \
    ExperimentSet, EVENT_TYPES
from baseline.utils import unzip_files, read_config_file
import shutil

__all__ = []
exporter = export(__all__)

from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
import sqlalchemy as sql
import sqlalchemy.orm as orm
from xpctl.backends.core import ExperimentRepo


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
        self.engine = sql.create_engine(uri, echo=False, paramstyle='format', pool_size=100)
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
        unpacked = client_experiment_to_put_result_consumable(exp)
        return self._put_result(task=task, config_obj=unpacked.config_obj, events_obj=unpacked.events_obj,
                                **unpacked.extra_args)
    
    def _put_result(self, task, config_obj, events_obj, **kwargs):
        session = self.Session()
        
        now = safe_get(kwargs, 'date', datetime.datetime.utcnow().isoformat())
        hostname = safe_get(kwargs, 'hostname', socket.gethostname())
        username = safe_get(kwargs, 'username', getpass.getuser())
        config_sha1 = safe_get(kwargs, 'sha1', hash_config(config_obj))
        label = safe_get(kwargs, 'label', get_experiment_label(config_obj, task, **kwargs))
        checkpoint = kwargs.get('checkpoint')
        version = safe_get(kwargs,  'version', __version__)
        dataset = safe_get(kwargs, 'dataset', config_obj.get('dataset'))
        date = safe_get(kwargs, 'exp_date', now)

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
            dataset=dataset,
            config=json.dumps(config_obj),
            hostname=hostname,
            username=username,
            date=date,
            version=version,
            status='CREATED',
            last_modified=now,
            events=events
        )
        try:
            session.add(experiment)
            session.commit()
            return BackendSuccess(message=experiment.eid)
        except sql.exc.SQLAlchemyError as e:
            return BackendError(message=str(e))
        
    def get_model_location(self, task, eid):
        session = self.Session()
        exp = session.query(SqlExperiment).get(eid)
        return BackendSuccess(exp.checkpoint)

    def update_prop(self, task, eid, prop, value):
        try:
            session = self.Session()
            exp = session.query(SqlExperiment).filter(SqlExperiment.task == task).filter(SqlExperiment.eid == eid)
            if exp is None or exp.scalar() is None:
                return BackendError(message='no experiment with id [{}] for task [{}]'.format(eid, task))
            # there must be a better way of getting a column value through a column name
            prev_value = getattr(session.query(SqlExperiment).get(eid), prop)
            session.query(SqlExperiment).filter(SqlExperiment.eid == eid).update({prop: value})
            session.commit()
            changed_value = getattr(session.query(SqlExperiment).get(eid), prop)
            return BackendSuccess(message='for experiment [{}] property [{}] was changed from [{}] to [{}].'
                                  .format(eid, prop, prev_value, changed_value))
        except sql.exc.SQLAlchemyError as e:
            return BackendError(message=str(e))

    def remove_experiment(self, task, eid):
        try:
            session = self.Session()
            exp = session.query(SqlExperiment).filter(SqlExperiment.task == task).filter(SqlExperiment.eid == eid)
            if exp is None or exp.scalar() is None:
                return BackendError(message='no experiment with id [{}] for task [{}]'.format(eid, task))
            model_loc_response = self.get_model_location(task, eid)
            model_loc = model_loc_response.message
            if model_loc is not None and type(model_loc_response) is not BackendError and os.path.exists(model_loc):
                try:
                    os.remove(model_loc)
                except IOError:
                    return BackendError(message='model {} exists on host but could not be removed'.format(model_loc))
            exp = session.query(SqlExperiment).get(eid)
            session.delete(exp)
            session.commit()
            try:
                assert session.query(SqlExperiment).get(eid) is None
                return BackendSuccess("record [{}] deleted successfully from database [{}]".format(eid, task))
            except AssertionError:
                return BackendError('delete failed: could not delete experiment {} from {} database'.format(eid, task))
        except sql.exc.SQLAlchemyError as e:
            return BackendError(message=str(e))

    def get_experiment_details(self, task, eid, event_type, metric):
        metrics = [x for x in listify(metric) if x.strip()]
        if event_type is None or event_type == 'None':
            event_type = 'test_events'
        session = self.Session()
        exp = session.query(SqlExperiment).filter(SqlExperiment.task == task).filter(SqlExperiment.eid == eid)
        if exp is None or exp.scalar() is None:
            return BackendError(message='no experiment with id [{}] for task [{}]'.format(eid, task))
        return self.sql_result_to_data_experiment(exp.one(), event_type, metrics)
    
    def get_results(self, task, prop, value, reduction_dim, metric, sort, numexp_reduction_dim, event_type):
        session = self.Session()
        metrics = [x for x in listify(metric) if x.strip()]
        if event_type is None or event_type == 'None':
            event_type = 'test_events'
        reduction_dim = reduction_dim if reduction_dim is not None else 'sha1'
        hits = session.query(SqlExperiment).filter(getattr(SqlExperiment, prop) == value)\
            .filter(SqlExperiment.task == task)
        if hits is None or not hits.first():
            return BackendError(message='no information available for [{}]: [{}] in task database [{}]'
                                .format(prop, value, task))
        data_experiments = []
        for exp in hits:
            data_experiment = self.sql_result_to_data_experiment(exp, event_type, metrics)
            if type(data_experiment) is BackendError:
                return data_experiment
            else:
                data_experiments.append(data_experiment)
        experiment_aggregate_set = self.aggregate_sql_results(data_experiments, reduction_dim, event_type,
                                                              numexp_reduction_dim)
        if sort is None or sort == 'None':
            return experiment_aggregate_set
        else:
            if event_type == 'test_events':
                if sort in METRICS_SORT_ASCENDING:
                    return experiment_aggregate_set.sort(sort, reverse=False)
                else:
                    return experiment_aggregate_set.sort(sort)
            else:
                return BackendError(message='experiments can only be sorted when event_type=test_events')
         
    def list_results(self, task, prop, value, user, metric, sort, event_type):
        session = self.Session()
        data_experiments = []
        if event_type is None or event_type == 'None':
            event_type = 'test_events'
        metrics = [x for x in listify(metric) if x.strip()]
        users = [x for x in listify(user) if x.strip()]
        if prop is None or prop == 'None':
            hits = session.query(SqlExperiment).filter(SqlExperiment.task == task)
        else:
            hits = session.query(SqlExperiment).filter(getattr(SqlExperiment, prop) == value). \
                   filter(SqlExperiment.task == task)
        if users:
            hits = hits.filter(SqlExperiment.username.in_(users))
        if hits.first() is None:
            return BackendError('No results in {} database for {} = {}'.format(task, prop, value))
        for exp in hits:
            data_experiment = self.sql_result_to_data_experiment(exp, event_type, metrics)
            if type(data_experiment) is BackendError:
                return data_experiment
            else:
                data_experiments.append(data_experiment)
        experiment_set = self.get_data_experiment_set(data_experiments)
        if sort is None or sort == 'None':
            return experiment_set
        else:
            if event_type == 'test_events':
                if sort in METRICS_SORT_ASCENDING:
                    return experiment_set.sort(sort, reverse=False)
                else:
                    return experiment_set.sort(sort)
            else:
                return BackendError(message='experiments can only be sorted when event_type=test_events')

    def find_experiments(self, task, prop, value):
        session = self.Session()
        sql_experiments = []
        hits = session.query(SqlExperiment).filter(getattr(SqlExperiment, prop) == value). \
            filter(SqlExperiment.task == task)
        if hits is None or not hits.first():
            return BackendError('No results in {} database for {} = {}'.format(task, prop, value))
        for exp in hits:
            sql_experiments.append(self.sql_result_to_data_experiment(exp, event_type='test_events', metrics_from_user=[]))
        return self.get_data_experiment_set(sql_experiments)
        
    def config2json(self, task, sha1):
        try:
            session = self.Session()
            config = session.query(SqlExperiment).filter(SqlExperiment.sha1 == sha1).one().config
            return json.loads(config)
        except sql.exc.SQLAlchemyError as e:
            return BackendError(message=str(e))

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
            return BackendError('could not get summary for task: [{}]'.format(task))
        return TaskDatasetSummarySet(task=task, data=store).groupby()

    def summary(self):
        tasks = self.get_task_names()
        if "system.indexes" in tasks:
            tasks.remove("system.indexes")
        return [self.task_summary(task) for task in tasks]
       
    def dump(self, zipfile='xpctl-sqldump-{}'.format(datetime.datetime.now().isoformat()), task_eids={}):
        """ dump reporting log and config for later consumption. you"""
        tasks = self.get_task_names() if not task_eids.keys() else list(task_eids.keys())
        session = self.Session()
        
        base_dir = '/tmp/xpctldump'
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)

        os.makedirs(base_dir, exist_ok=True)
        for task in tasks:
            _dir = os.path.join(base_dir, task)
            os.makedirs(_dir)
            sql_exps = session.query(SqlExperiment).filter(SqlExperiment.task == task)
            for sql_exp in sql_exps:
                exp = self.sql_result_to_data_experiment(sql_exp, event_type=[], metrics_from_user=[])
                if type(exp) is BackendError:
                    print(exp.message)
                else:
                    write_experiment(exp, _dir)
        return shutil.make_archive(base_name=zipfile, format='zip', root_dir='/tmp', base_dir='xpctldump')

    def restore(self, dump):
        """ if dump is in zip format, will unzip it. expects the following dir structure in the unzipped file:
        <root>
         - <task>
           - <id>
             - <id>-reporting.log
             - <id>-config.yml
             - <id>-meta.yml (any meta info such as label, username etc.)
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

    def aggregate_sql_results(self, data_experiments, reduction_dim, event_type, numexp_reduction_dim):
        experiment_set = self.get_data_experiment_set(data_experiments)
        return aggregate_results(experiment_set, reduction_dim, event_type, numexp_reduction_dim)

    def sql_result_to_data_experiment(self, exp, event_type, metrics_from_user):
        _exp = Experiment(
            task=exp.task,
            eid=exp.eid,
            username=exp.username,
            hostname=exp.hostname,
            config=exp.config,
            exp_date=exp.date,
            label=exp.label,
            dataset=exp.dataset,
            sha1=exp.sha1,
            version=exp.version,
            train_events=[],
            valid_events=[],
            test_events=[]
        )
        event_types = [event_type] if event_type else EVENT_TYPES
        for event_type in event_types:
            phase = self.event2phase(event_type)
            if type(phase) is BackendError:
                return phase
            phase_events = [event for event in exp.events if event.phase == phase]
            if len(phase_events) == 0:
                continue
            metrics = self.get_filtered_metrics(self.get_sql_metrics(phase_events[0]), set(metrics_from_user))
            if type(metrics) is BackendError:
                return metrics
            results = self.flatten([self.create_results(event, metrics) for event in phase_events])
            for r in results:
                _exp.add_result(r, event_type)
        return _exp

    @staticmethod
    def event2phase(event_type):
        if event_type == 'train_events':
            return 'Train'
        if event_type == 'valid_events':
            return 'Valid'
        if event_type == 'test_events':
            return 'Test'
        BackendError(message='Unknown event type {}'.format(event_type))
    
    @staticmethod
    def get_filtered_metrics(metrics_from_db, metrics_from_user):
        if not metrics_from_user:
            metrics = list(metrics_from_db)
        elif metrics_from_user - metrics_from_db:
            return BackendError(message='Metrics [{}] not found'.format(','.join(list(metrics_from_user - metrics_from_db))))
        else:
            metrics = list(metrics_from_user)
        return metrics
    
    @staticmethod
    def get_sql_metrics(event):
        return set([r.metric for r in event.results])
    
    @staticmethod
    def flatten(_list):
        return [item for sublist in _list for item in sublist]
    
    @staticmethod
    def create_results(event, metrics):
        results = []
        for r in event.results:
            if r.metric in metrics:
                results.append(
                    Result(metric=r.metric, value=r.value, tick_type=event.tick_type, tick=event.tick, phase=event.phase)
                )
        return results

    @staticmethod
    def get_data_experiment_set(data_experiments):
        return ExperimentSet(data_experiments)


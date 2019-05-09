from backends.data import Experiment, ExperimentSet, Result, Error
TRAIN_EVENT = 'train_events'
VALID_EVENT = 'valid_events'
TEST_EVENT = 'test_events'
EVENT_TYPES = [TRAIN_EVENT, VALID_EVENT, TEST_EVENT]


class MongoResult(object):
    """ a result data point"""
    def __init__(self, metric, value, task, eid, username, hostname, label, config, dataset, date, sha1, event_type,
                 tick_type, tick, phase, version):
        super(MongoResult, self).__init__()
        self.metric = metric
        self.value = value
        self.task = task
        self.eid = eid
        self.username = username
        self.hostname = hostname
        self.label = label
        self.dataset = dataset
        self.date = date
        self.sha1 = sha1
        self.event_type = event_type
        self.config = config
        self.tick_type = tick_type
        self.tick = tick
        self.phase = phase
        self.version = version
    
    def get_prop(self, field):
        return self.__dict__[field]


class MongoResultSet(object):
    """ a list of result objects"""
    def __init__(self, data):
        super(MongoResultSet, self).__init__()
        self.data = data if data else []
        self.length = len(self.data)
    
    def add_data(self, data_point):
        """
        add a result data point
        :param data_point:
        :return:
        """
        self.data.append(data_point)
        self.length += 1
    
    # TODO: add property annotations
    def __getitem__(self, i):
        return self.data[i]
    
    def __iter__(self):
        for i in range(self.length):
            yield self.data[i]
    
    def __len__(self):
        return self.length
    
    def groupby(self, key):
        """ group the data points by key"""
        data_groups = {}
        for datum in self.data:
            field = datum.get_prop(key)
            if field not in data_groups:
                data_groups[field] = [datum]
            else:
                data_groups[field].append(datum)
        return data_groups
    
    def experiments(self):
        grouped_results = self.groupby('eid')
        experiments = []
        for eid, resultset in grouped_results.items():
            first_result = resultset[0]
            task = first_result.task
            eid = str(first_result.eid)
            username = first_result.username
            hostname = first_result.hostname
            label = first_result.label
            dataset = first_result.dataset
            date = first_result.date
            sha1 = first_result.sha1
            config = first_result.config
            version = first_result.version
            exp = Experiment(
                             task=task,
                             eid=eid,
                             sha1=sha1,
                             config=config,
                             dataset=dataset,
                             username=username,
                             hostname=hostname,
                             exp_date=date,
                             label=label,
                             version=version,
                             train_events=[],
                             valid_events=[],
                             test_events=[])
            for _result in resultset:
                r = Result(metric=_result.metric, value=_result.value, tick_type=_result.tick_type, tick=_result.tick,
                           phase=_result.phase)
                exp.add_result(r, _result.event_type)
            experiments.append(exp)
        return ExperimentSet(experiments)


def get_metrics_mongo(xs):
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


def mongo_to_experiment_set(task, all_results, event_type, metrics):
    data = []
    event_types = [event_type] if event_type else EVENT_TYPES
    metrics_from_user = set([x for x in metrics if x.strip()])
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
            metrics_from_db = get_metrics_mongo(result[event_type])
            if not metrics_from_user:
                metrics = list(metrics_from_db)
            elif metrics_from_user - metrics_from_db:
                return Error(message='Metrics [{}] not found for experiment [{}] in [{}] database'.format(','.join(
                    list(metrics_from_user - metrics_from_db)), _id, task))
            else:
                metrics = list(metrics_from_user)
            # for train_events we can have different metrics than test_events
            for record in result[event_type]:  # train_event epoch 0,
                for metric in metrics:
                    data.append(MongoResult(
                        metric=metric,
                        value=record[metric],
                        task=task,
                        eid=str(_id),
                        username=username,
                        hostname=hostname,
                        label=label,
                        config=config,
                        dataset=dataset,
                        date=date,
                        sha1=sha1,
                        event_type=event_type,
                        tick_type=record['tick_type'],
                        tick=record['tick'],
                        phase=record['phase'],
                        version=version
                    ))
    if not data:
        return Error(message='No results from the query')
    rs = MongoResultSet(data=data)
    return rs.experiments()

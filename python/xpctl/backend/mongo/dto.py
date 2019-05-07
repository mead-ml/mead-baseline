from xpctl.data import Experiment, ExperimentSet, TaskDatasetSummary, TaskDatasetSummarySet, Result
TRAIN_EVENT = 'train_events'
DEV_EVENT = 'valid_events'
TEST_EVENT = 'test_events'


class MongoResult(object):
    """ a result data point"""
    def __init__(self, metric, value, task, _id, username, hostname, label, config, dataset, date, sha1, event_type, epoch,
                 version):
        super(MongoResult, self).__init__()
        self.metric = metric
        self.value = value
        self.task = task
        self._id = _id
        self.username = username
        self.hostname = hostname
        self.label = label
        self.dataset = dataset
        self.date = date
        self.sha1 = sha1
        self.event_type = event_type
        self.config = config
        self.epoch = epoch
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
        grouped_results = self.groupby('_id')
        experiments = []
        for _id, resultset in grouped_results.items():
            first_result = resultset[0]
            task = first_result.task
            _id = str(first_result._id)
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
                             _id=_id,
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
                r = Result(metric=_result.metric, value=_result.value, epoch=_result.epoch)
                exp.add_result(r, _result.event_type)
            experiments.append(exp)
        return ExperimentSet(experiments)


class MongoTaskDatasetSummary(TaskDatasetSummary):
    def __init__(self, task, dataset, experiment_set):
        super(MongoTaskDatasetSummary, self).__init__(task, dataset, experiment_set)


class MongoTaskDatasetSummarySet(TaskDatasetSummarySet):
    def __init__(self, task, data):
        super(MongoTaskDatasetSummarySet, self).__init__(task, data)
    

class MongoError(object):
    def __init__(self, message, code=550):
        super(MongoError, self).__init__()
        self.message = message
        self.code = code


class MongoSuccess(object):
    def __init__(self, message, code=250):
        super(MongoSuccess, self).__init__()
        self.message = message
        self.code = code

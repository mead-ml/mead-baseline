from xpctl.data import Experiment, result, ExperimentSet


class MongoResult(object):
    """ a result data point"""
    def __init__(self, metric, value, _id, username, label, dataset, date, sha1, event_type, epoch):
        super(MongoResult, self).__init__()
        self.metric = metric
        self.value = value
        self._id = _id
        self.username = username
        self.label = label
        self.dataset = dataset
        self.date = date
        self.sha1 = sha1
        self.event_type = event_type,
        self.epoch = epoch
    
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
            _id = _id
            username = first_result.username
            label = first_result.label
            dataset = first_result.dataset
            date = first_result.date
            sha1 = first_result.sha1
            exp = Experiment(train_results=[], dev_results=[], test_results=[], _id=_id, username=username,
                             label=label, dataset=dataset, date=date, sha1=sha1)
            for _result in resultset:
                exp.add_result(result=result(metric=_result.metric, value=_result.value, epoch=0),
                               event_type=_result.event_type)
            experiments.append(exp)
        return ExperimentSet(data=experiments)



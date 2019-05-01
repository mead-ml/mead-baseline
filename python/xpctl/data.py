from copy import deepcopy


class Result(object):
    """ a result data point"""
    def __init__(self, metric, value, _id, username, label, dataset, date, sha1, event_type):
        super(Result, self).__init__()
        self.metric = metric
        self.value = value
        self._id = _id
        self.username = username
        self.label = label
        self.dataset = dataset
        self.date = date
        self.sha1 = sha1
        self.event_type = event_type
    
    def get_prop(self, field):
        return self.__dict__[field]


class AggregateResult(object):
    """ a result data point"""
    def __init__(self, metric, values, **kwargs):
        super(AggregateResult, self).__init__()
        self.metric = metric
        self.values = values # {'min': something, 'avg': something ..}
        self._id = kwargs.get('_id')
        self.username = kwargs.get('username')
        self.label = kwargs.get('label')
        self.dataset = kwargs.get('dataset')
        self.date = kwargs.get('date')
        self.sha1 = kwargs.get('sha1')
        self.event_type = kwargs.get('event_type')
    
    def get_prop(self, field):
        return self.__dict__[field]

# TODO: add property annotations
class AggregateResultSet(object):
    """ a list of result objects"""
    def __init__(self, data):
        super(AggregateResultSet, self).__init__()
        self.data = data if data else []
        self.length = len(self.data)
    
    def add_data(self, data_point):
        """
        add a aggregateresult data point
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


class ResultSet(object):
    """ a list of result objects"""
    def __init__(self, data):
        super(ResultSet, self).__init__()
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
                data_groups[field] = ResultSet([datum])
            else:
                data_groups[field].add_data(datum)
        return ResultSetGroup(data_groups, key)


class ResultSetGroup(object):
    """ a group of resultset objects"""
    def __init__(self, grouped_resultsets, grouping_key):
        self.resultsets = grouped_resultsets
        self.grouping_key = grouping_key
        self.length = len(grouped_resultsets)

    def reduce(self, aggregate_fns):
        """ aggregate results across a result group"""
        data = {}
        for prop_value, resultset in self.resultsets.items():
            # one resultset has n result(s),
            data[prop_value] = {}
            for datum in resultset.data:
                if datum.metric not in data[prop_value]:
                    data[prop_value][datum.metric] = [datum.value]
                else:
                    data[prop_value][datum.metric].append(datum.value)
        aggregate_resultset = AggregateResultSet(data=[])
        for prop_value in data:
            values = {}
            for metric in data[prop_value]:
                for fn_name, fn in aggregate_fns.items():
                    agg_value = fn(data[prop_value][metric])
                    values[fn_name] = agg_value
                d = {self.grouping_key: prop_value}
                agr = deepcopy(AggregateResult(metric=metric, values=values, **d))
                aggregate_resultset.add_data(agr)
        return aggregate_resultset

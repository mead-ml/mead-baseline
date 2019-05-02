from copy import deepcopy
from collections import namedtuple

result = namedtuple('result', ['metric', 'value', 'epoch'], verbose=True)
aggregate_result = namedtuple('aggregate_result', ['metric', 'values'], verbose=True)
TRAIN_EVENT = 'train_events'
DEV_EVENT = 'valid_events'
TEST_EVENT = 'test_events'


class Experiment(object):
    """ an experiment"""
    def __init__(self, train_results, dev_results, test_results, _id, username, label, dataset, date, sha1):
        super(Experiment, self).__init__()
        self.train_results = set(train_results)
        self.dev_results = set(dev_results)
        self.test_results = set(test_results)
        self._id = _id
        self.username = username
        self.label = label
        self.dataset = dataset
        self.date = date
        self.sha1 = sha1
    
    def get_prop(self, field):
        if field not in self.__dict__:
            raise ValueError('{} does not have a property {}'.format(self.__class__, field))
        return self.__dict__[field]
    
    def add_result(self, result, event_type):
        if event_type == TRAIN_EVENT:
            self.train_results.add(result)
        elif event_type == DEV_EVENT:
            self.dev_results.add(result)
        elif event_type == TEST_EVENT:
            self.test_results.add(result)
        else:
            raise NotImplementedError('no handler for event type: [{}]'.format(event_type))


class ExperimentSet(object):
    """ a list of experiment objects"""
    def __init__(self, data):
        super(ExperimentSet, self).__init__()
        self.data = data if data else []  # this should ideally be a set but the items are not hashable
        self.length = len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    def __iter__(self):
        for i in range(self.length):
            yield self.data[i]
    
    def __len__(self):
        return self.length

    def add_data(self, datum):
        """
        add a experiment data point
        :param datum:
        :return:
        """
        self.data.append(datum)
        self.length += 1
        
    def groupby(self, key):
        """ group the data points by key"""
        data_groups = {}
        for datum in self.data:
            field = datum.get_prop(key)
            if field not in data_groups:
                data_groups[field] = ExperimentSet([datum])
            else:
                data_groups[field].add_data(datum)
        return ExperimentGroup(data_groups, key)


class ExperimentGroup(object):
    """ a group of resultset objects"""
    def __init__(self, grouped_experiments, grouping_key):
        super(ExperimentGroup, self).__init__()
        self.grouped_experiments = grouped_experiments
        self.grouping_key = grouping_key
    
    def items(self):
        return self.grouped_experiments.items()
    
    def keys(self):
        return self.grouped_experiments.keys()
    
    def values(self):
        return self.grouped_experiments.values()
    
    def __iter__(self):
        for k, v in self.grouped_experiments.items():
            yield (k, v)
    
    def get(self, key):
        return self.grouped_experiments.get(key)
    
    def __len__(self):
        return len(self.grouped_experiments.keys())
    
    def reduce(self, aggregate_fns, event_type=TEST_EVENT):
        """ aggregate results across a result group"""
        data = {}
        f_map = {TRAIN_EVENT: 'train_results', DEV_EVENT: 'dev_results', TEST_EVENT: 'test_results'}
        for prop_value, experiments in self.grouped_experiments.items():
            # one resultset has n result(s),
            data[prop_value] = {}
            for experiment in experiments:
                results = experiment.get_prop(f_map[event_type])
                for result in results:
                    if result.metric not in data[prop_value]:
                        data[prop_value][result.metric] = [result.value]
                    else:
                        data[prop_value][result.metric].append(result.value)
        aggregate_resultset = ExperimentAggregateSet(data=[])
        for prop_value in data:
            values = {}
            d = {self.grouping_key: prop_value}
            agr = deepcopy(ExpermentAggregate(**d))
            for metric in data[prop_value]:
                for fn_name, fn in aggregate_fns.items():
                    agg_value = fn(data[prop_value][metric])
                    values[fn_name] = agg_value
                agr.add_result(deepcopy(aggregate_result(metric=metric, values=values)), event_type=event_type)
            aggregate_resultset.add_data(agr)
        return aggregate_resultset
    
    def trim(self, num_elements):
        """for each group in the resultsets, trim to num_elements"""
        keys = self.keys()
        to_pop = list(keys)[num_elements:]
        for key in to_pop:
            self.grouped_experiments.pop(key)


class ExpermentAggregate(object):
    """ a result data point"""
    def __init__(self, train_results=[], dev_results=[], test_results=[], **kwargs):
        super(ExpermentAggregate, self).__init__()
        self.train_results = train_results
        self.dev_results = dev_results
        self.test_results = test_results
        self._id = kwargs.get('_id')
        self.username = kwargs.get('username')
        self.label = kwargs.get('label')
        self.dataset = kwargs.get('dataset')
        self.date = kwargs.get('date')
        self.sha1 = kwargs.get('sha1')
    
    def get_prop(self, field):
        return self.__dict__[field]

    def add_result(self, result, event_type):
        if event_type == TRAIN_EVENT:
            self.train_results.append(result)
        elif event_type == DEV_EVENT:
            self.dev_results.append(result)
        elif event_type == TEST_EVENT:
            self.test_results.append(result)
        else:
            raise NotImplementedError('no handler for event type: [{}]'.format(event_type))


class ExperimentAggregateSet(object):
    """ a list of aggregate result objects"""
    def __init__(self, data):
        super(ExperimentAggregateSet, self).__init__()
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

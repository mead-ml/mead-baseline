from copy import deepcopy

TRAIN_EVENT = 'train_events'
DEV_EVENT = 'valid_events'
TEST_EVENT = 'test_events'


class Result(object):
    def __init__(self, metric, value, epoch):
        super(Result, self).__init__()
        self.metric = metric
        self.value = value
        self.epoch = epoch

    def get_prop(self, field):
        if field not in self.__dict__:
            raise ValueError('{} does not have a property {}'.format(self.__class__, field))
        return self.__dict__[field]
        
        
class AggregateResult(object):
    def __init__(self, metric, values):
        super(AggregateResult, self).__init__()
        self.metric = metric
        self.values = values

    def get_prop(self, field):
        if field not in self.__dict__:
            raise ValueError('{} does not have a property {}'.format(self.__class__, field))
        return self.__dict__[field]


class Experiment(object):
    """ an experiment"""
    def __init__(self, train_events, valid_events, test_events, task, _id, username, hostname, config, exp_date, label, dataset,
                 sha1, version):
        super(Experiment, self).__init__()
        self.task = task
        self.train_events = train_events if train_events is not None else []
        self.valid_events = valid_events if valid_events is not None else []
        self.test_events = test_events if test_events is not None else []
        self.eid = _id
        self.username = username
        self.hostname = hostname
        self.config = config
        self.exp_date = exp_date
        self.label = label
        self.dataset = dataset
        self.exp_date = exp_date
        self.sha1 = sha1
        self.version = version
    
    def get_prop(self, field):
        if field not in self.__dict__:
            raise ValueError('{} does not have a property {}'.format(self.__class__, field))
        return self.__dict__[field]
    
    def add_result(self, result, event_type):
        if event_type == TRAIN_EVENT:
            self.train_events.append(result)
        elif event_type == DEV_EVENT:
            self.valid_events.append(result)
        elif event_type == TEST_EVENT:
            self.test_events.append(result)
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
        if len(self.data) == 0:
            raise RuntimeError('Trying to group empty experiment set')
        task = self.data[0].get_prop('task')
        for datum in self.data:
            if datum.get_prop('task') != task:
                raise RuntimeError('Should not be grouping two experiments from different tasks')
            field = datum.get_prop(key)
            if field not in data_groups:
                data_groups[field] = ExperimentSet([datum])
            else:
                data_groups[field].add_data(datum)
        return ExperimentGroup(data_groups, key, task)
    
    def sort(self, key, reverse=True):
        """
        you can only sort when event_type is test, because there is only one data point
        :param key: metric to sort on
        :param reverse: reverse=True always except when key is avg_loss
        :return:
        """
        if key is None:
            return self
        test_results = [(index, [y for y in x.get_prop(TEST_EVENT) if y.metric == key][0]) for index, x in
                        enumerate(self.data)]
        test_results.sort(key=lambda x: x[1].value, reverse=reverse)
        final_results = []
        for index, _ in test_results:
            final_results.append(self.data[index])
        return ExperimentSet(data=final_results)


class ExperimentGroup(object):
    """ a group of resultset objects"""
    def __init__(self, grouped_experiments, reduction_dim, task):
        super(ExperimentGroup, self).__init__()
        self.grouped_experiments = grouped_experiments
        self.reduction_dim = reduction_dim
        self.task = task
    
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
        num_experiments = {}
        for reduction_dim_value, experiments in self.grouped_experiments.items():
            num_experiments[reduction_dim_value] = len(experiments)
            data[reduction_dim_value] = {}
            for experiment in experiments:
                results = experiment.get_prop(event_type)
                for result in results:
                    if result.metric not in data[reduction_dim_value]:
                        data[reduction_dim_value][result.metric] = [result.value]
                    else:
                        data[reduction_dim_value][result.metric].append(result.value)
        # for each reduction dim value, (say when sha1 = x), all data[x][metric] lists should have the same length.
        for reduction_dim_value in data:
            lengths = []
            for metric in data[reduction_dim_value]:
                lengths.append(len(data[reduction_dim_value][metric]))
            try:
                assert len(set(lengths)) == 1
            except AssertionError:
                raise AssertionError('when reducing experiments over {}, for {}={}, the number of results are not the '
                                     'same over all metrics'.format(self.reduction_dim, self.reduction_dim,
                                                                    reduction_dim_value))
            
        aggregate_resultset = ExperimentAggregateSet(data=[])
        for reduction_dim_value in data:
            values = {}
            d = {self.reduction_dim: reduction_dim_value, 'num_exps': num_experiments[reduction_dim_value]}
            agr = deepcopy(ExperimentAggregate(task=self.task, **d))
            for metric in data[reduction_dim_value]:
                for fn_name, fn in aggregate_fns.items():
                    agg_value = fn(data[reduction_dim_value][metric])
                    values[fn_name] = agg_value
                agr.add_result(deepcopy(AggregateResult(metric=metric, values=values)), event_type=event_type)
            aggregate_resultset.add_data(agr)
        return aggregate_resultset
    

class ExperimentAggregate(object):
    """ a result data point"""
    def __init__(self, task, train_events=[], valid_events=[], test_events=[], **kwargs):
        super(ExperimentAggregate, self).__init__()
        self.train_events = train_events if train_events is not None else []
        self.valid_events = valid_events if valid_events is not None else []
        self.test_events = test_events if test_events is not None else []
        self.task = task
        self.num_exps = kwargs.get('num_exps')
        self.eid = kwargs.get('eid')
        self.username = kwargs.get('username')
        self.label = kwargs.get('label')
        self.dataset = kwargs.get('dataset')
        self.exp_date = kwargs.get('exp_date')
        self.sha1 = kwargs.get('sha1')
    
    def get_prop(self, field):
        return self.__dict__[field]

    def add_result(self, aggregate_result, event_type):
        if event_type == TRAIN_EVENT:
            self.train_events.append(aggregate_result)
        elif event_type == DEV_EVENT:
            self.valid_events.append(aggregate_result)
        elif event_type == TEST_EVENT:
            self.test_events.append(aggregate_result)
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

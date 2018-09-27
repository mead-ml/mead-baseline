from __future__ import absolute_import, division, print_function, unicode_literals
import six
from six.moves import cPickle as pickle

import os
import math
import time
import platform
import functools
from enum import Enum
from pprint import pformat
from collections import defaultdict
from multiprocessing.managers import BaseManager
import numpy as np
from baseline.utils import export as exporter
from baseline.utils import hash_config
from hpctl.utils import Label


__all__ = []
export = exporter(__all__)

dd_list = functools.partial(defaultdict, list)
ddd_list = functools.partial(defaultdict, dd_list)
dddd_list = functools.partial(defaultdict, ddd_list)


@six.python_2_unicode_compatible
class States(Enum):
    DONE = '\u2714'
    KILLED = '\u2717'
    RUNNING = '\u21bb'
    WAITING = '\u231b'
    UNKNOWN = '?'

    def __str__(self):
        return self.value


@export
class Results(object):
    """An object that aggregates results from jobs.

    DataStructure:
        Dict: {
            str, exp_hash : Dict[
                hpctl.utils.Label: Dict [
                    str, phase : Dict [
                        str, log field: List[float]
                    ]
                ],
                'state': hpctl.results.States, The state of that job.
                'timestamp': int, When that job started.
            ]
        }

    Data is results in a columnish results where most entries are a list that
    represent a timeseries. This lets use to easy `OLAP`-ish queries rather
    than tracking the best performance in the frontend.
    """
    def __init__(self):
        super(Results, self).__init__()
        self.results = defaultdict(dddd_list)
        self.label_to_config = {}
        self.label_to_name = defaultdict(list)
        self.name_to_label = defaultdict(list)
        self.exp_to_config = {}

    def add_experiment(self, exp_config):
        exp_hash = hash_config(exp_config)
        self.exp_to_config[exp_hash] = exp_config

    def get_experiment_config(self, exp_hash):
        return self.exp_to_config.get(exp_hash, {})

    @classmethod
    def create(cls, file_name="results", **kwargs):
        """Load a results if found, create otherwise.

        :param exp: str, The name of the experiment.
        :param mead_hash: str, The name of the results.

        :returns:
            hpctl.results.Results
        """
        manager = ResultsManager()
        manager.start()
        if file_name is None:
            return manager.Results()
        file_name = file_name + '.p'
        if not os.path.exists(file_name):
            return manager.Results()
        s = pickle.load(open(file_name, 'rb'))
        for exp in s.results.values():
            for v in exp.values():
                # If you are loading a results and a job was running is dead now.
                if v['state'] is States.RUNNING:
                    v['state'] = States.KILLED
        results = manager.Results()
        results.restore(s)
        return results

    def restore(self, results):
        """Copy data from one results to another, used to populate the manager fro pickle reload."""
        self.results = results.results
        self.label_to_config = results.label_to_config
        self.label_to_name = results.label_to_name
        self.name_to_label = results.name_to_label
        self.exp_to_config = results.exp_to_config

    def save(self, file_name="results"):
        """Persist the results.

        :param exp: str, The name of the experiment.
        :param mead_hash: str, The name of the results.
        """
        file_name = file_name + ".p"
        pickle.dump(self, open(file_name, 'wb'))

    def insert(self, label, config):
        """Add a new entry to the results.

        :param label: str, The sha1 of the config.
        :param human: str, The human name of the config.
        :param config: dict, The config.
        """
        self.label_to_config[label] = config
        self.results[label.exp][label]['time_stamp'] = time.time()
        self.set_waiting(label)
        self.label_to_name[label.sha1].append(label.name)
        self.name_to_label[label.name].append(label.sha1)

    def update(self, label, message):
        """Update entry with a new log info.

        :param label: str, The label for the log (a sha1 of a config).
        :param message: dict, The log info.
        """
        phase = message.pop('phase')
        if phase == 'Test':
            self.results[label.exp][label]['state'] = States.DONE
        for k, v in message.items():
            self.results[label.exp][label][phase][k].append(v)

    def get_config(self, label):
        """Get the config from the sha1.

        :param label: str, The sha1

        :returns:
            dict, The config.
        """
        return search(label, self.label_to_config, prefix=False)

    def get_recent(self, label, phase, metric):
        """Get the most recent entry for the somethings

        :param label: str, The config label to look up.
        :param phase: str, The phase of training to look at.
        :param metric: str, The metric to look for.

        :returns:
            The last value in the column or 0.0
        """
        res = self.results[label.exp][label][phase][metric]
        res = res[-1] if res else 0.0
        return res

    def get_best(self, label, phase, metric):
        """Get the best performance of a given label for a given metric.

        :param label: str, The label of the model.
        :param phase: str, The name of the phase.
        :param metric: str, The metric to look up.

        :returns: tuple (float, int)
            [0]: The value the best model achieved.
            [1]: The tick value the best results was got at.
        """
        data = self.results[label.exp][label][phase][metric]
        if not data:
            return 0.0, 0.0
        val = np.max(data)
        idx = self.results[label.exp][label][phase]['tick'][np.argmax(data)]
        return val, idx

    def find_best(self, exp_hash, phase, metric):
        """Get the best performance for a given metric.

        :param phase: str, The name of the phase.
        :param metric: str, The metric to look up.

        :returns: tuple (str, float, int)
            [0]: The sha1 label of the best performing model.
            [1]: The value the best model achieved.
            [2]: The tick value the best results was got at.
        """
        best_label = None
        best_val = 0
        best_idx = None
        for label in self.results[exp_hash]:
            data = self.results[exp_hash][label][phase][metric]
            if not data:
                continue
            val = np.max(data)
            idx = self.results[exp_hash][label][phase]['tick'][np.argmax(data)]
            if val > best_val:
                best_val = val
                best_label = label
                best_idx = idx
        return best_label, best_val, best_idx

    def get_best_per_label(self, exp_hash, phase, metric):
        """Get the best performance for a given metric across all labels.

        :param phase: str, The name of the phase.
        :param metric: str, The metric to look up.

        :returns: tuple (List[str], List[float], List[int])
            [0]: The sha1 label of the best performing model.
            [1]: The value the best model achieved.
            [2]: The tick value the best results was got at.
        """
        labels = self.get_labels(exp_hash)
        vals = []
        idxs = []
        for label in labels:
            val, idx = self.get_best(label, phase, metric)
            vals.append(val)
            idxs.append(idx)
        return labels, vals, idxs

    def get_labels(self, exp_hash):
        """Get all the labels in the data results.

        :returns:
            List[str]: The list of labels sorted by the time they started.
        """
        labels = [(x, self.results[exp_hash][x]['time_stamp']) for x in self.results[exp_hash]]
        labels = sorted(labels, key=lambda x: x[1])
        return [l[0] for l in labels]

    def get_experiments(self):
        return [x for x in self.results]

    def get_human(self, label):
        """Get the human label from the sha1.

        :param label: str, The sha1

        :returns:
            str, The human label.
        """
        return search(label, self.label_to_name, prefix=False)


    def get_human_prefix(self, label):
        """Get the human label from the sha1 with a prefix search, 'ab' matches 'abc'

        :param label: str, The sha1

        :returns:
            str, The human label.
        """
        return search(label, self.label_to_name, prefix=True)

    def get_label(self, label):
        """Get the sha1 from the human label.

        :param label: str, The human label

        :returns:
            str, The sha1.
        """
        return search(label, self.name_to_label, prefix=False)

    def get_label_prefix(self, label):
        """Get the sha1 from the human label with a prefix search, 'ab' matches 'abc'

        :param label: str, The sha1

        :returns:
            str, The human label.
        """
        return search(label, self.name_to_label, prefix=True)

    def get_state(self, label):
        """Get the job state based on the label.

        :param label: hpctl.utils.Label, The sha1 of the config.

        :returns:
            str, The unicode of the status.
        """
        state = self.results[label.exp][label]['state']
        if state not in States:
            return States.UNKNOWN
        return state

    def set_state(self, label, state):
        """Set the state of a job.

        :param label: hpctl.utils.Label, The label to set the state on.
        :param state: hpctl.results.States, The sate to set for the job.
        """
        self.results[label.exp][label]['state'] = state

    def set_killed(self, label):
        """Set a job to killed.

        :param label: hpctl.utils.Label, The label to set as killed.
        """
        self.set_state(label, States.KILLED)

    def set_waiting(self, label):
        self.set_state(label, States.WAITING)

    def set_running(self, label):
        self.set_state(label, States.RUNNING)

    def __str__(self):
        return pformat(self.results)

    def __del__(self):
        for exp in self.results:
            for label in self.results[exp]:
                state = self.get_state(label)
                if state == States.RUNNING or state == States.WAITING:
                    self.set_killed(label)


class BaseResults(object):
    def __init__(self):
        super(BaseResults, self).__init__()

    @classmethod
    def create(cls, **kwargs):
        pass

    def save(self, file_name=None):
        pass

    def insert(self, label, config):
        pass

    def update(self, label, message):
        pass

    def get_config(self, label):
        pass

    def get_recent(self, label, phase, metric):
        pass

    def get_best(self, label, phase, metric):
        pass

    def find_best(self, exp_hash, phase, metric):
        pass

    def get_best_per_label(self, exp_hash, phase, metric):
        pass

    def get_labels(self, exp_hash):
        pass

    def get_experiments(self):
        pass

    def get_state(self, label):
        pass

    def set_state(self, label, state):
        pass

    def set_killed(self, label):
        pass

    def set_waiting(self, label):
        pass

    def set_running(self, label):
        pass

    def command(self):
        pass

# Create the results as a multiprocessing manager so that we can share the
# results across processes.
class ResultsManager(BaseManager):
    pass
ResultsManager.register(str('Results'), Results)


def search(key, table, prefix=True):
    """Search for `key` in `table`.

    :param key: str, The string to look for.
    :param table: dic, The table to look in.
    :param prefix: bool, So a prefix search.

    :returns:
        Value in table or None
        if it is a prefix search it returns the full key and the value or (None, None)
    """
    if key in table:
        if prefix:
            return key, table[key]
        return table[key]
    if prefix:
        for k, v in table.items():
            if k.startswith(key):
                return k, v
    if prefix:
        return None, None
    return None


def get_results(results_config):
    kind = results_config.pop('type', 'local')
    if kind == 'remote':
        from hpctl.remote import RemoteResults
        return RemoteResults(**results_config)
    return Results.create(**results_config)

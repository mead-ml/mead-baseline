from __future__ import absolute_import, division, print_function, unicode_literals
import six

import requests
import cachetools
from baseline.utils import export as exporter
from xpctl.core import ExperimentRepo
from hpctl.utils import Label
from hpctl.backend import Backend
from hpctl.results import Results, States


__all__ = []
export = exporter(__all__)


def _get(url):
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception("Failed GET on {}".format(url))
    return r.json()


@export
class RemoteBackend(Backend):
    def __init__(self, host, port, **kwargs):
        super(RemoteBackend, self).__init__()
        self.labels = []
        self.url = 'http://{host}:{port}/hpctl/v1'.format(host=host, port=int(port))

    def any_done(self):
        return True

    def all_done(self):
        # Track all label you personally launched and check if they are done.
        undone = []
        for label in self.labels:
            state = _get(
                "{url}/state/{exp}/{sha1}/{name}".format(url=self.url, **label)
            )['state']
            state = state.encode('utf-8') if six.PY2 else state
            if state != str(States.DONE) and state != str(States.KILLED):
                undone.append(label)
        self.labels = undone
        return not self.labels

    def launch(self, **kwargs):
        kwargs['command'] = 'launch'
        self.labels.append(kwargs['label'])
        kwargs['label'] = str(kwargs['label'])
        r = requests.post("{}/launch".format(self.url), json=kwargs)
        if r.status_code != 200:
            raise Exception


@export
class RemoteResults(Results):
    """Interact with the results object via the flask frontend."""
    def __init__(self, host='localhost', port=5000, cache_time=15, **kwargs):
        super(RemoteResults, self).__init__()
        self.url = 'http://{host}:{port}/hpctl/v1'.format(host=host, port=port)
        cache = cachetools.TTLCache(maxsize=1000, ttl=cache_time)
        self.get = cachetools.cached(cache)(_get)

    def add_experiment(self, exp_config):
        requests.post("{url}/experiment/add".format(url=self.url), json=exp_config)

    def get_experiment_config(self, exp_hash):
        return self.get("{url}/config/{exp}".format(url=self.url, exp=exp_hash))

    def get_experiments(self):
        resp = self.get("{url}/experiments".format(url=self.url))
        return resp['experiments']

    def get_labels(self, exp_hash):
        resp = self.get("{url}/labels/{exp}".format(url=self.url, exp=exp_hash))
        labels = []
        for res in resp:
            labels.append(Label(exp_hash, res['sha1'], res['name']))
        return labels

    def get_state(self, label):
        resp = self.get("{url}/state/{exp}/{sha1}/{name}".format(url=self.url, **label))
        state = resp['state']
        return state.encode('utf-8') if six.PY2 else state

    def get_recent(self, label, phase, metric):
        resp = self.get(
            "{url}/result/recent/{exp}/{sha1}/{name}/{phase}/{metric}".format(
                url = self.url, phase=phase, metric=metric, **label
            )
        )
        return resp['value']

    def find_best(self, exp, phase, metric):
        resp = self.get(
            "{url}/result/best/{exp}/{phase}/{metric}".format(
                url = self.url, exp=exp, phase=phase, metric=metric
            )
        )
        return Label.parse(resp['label']), resp['value'], resp['step']

    def get_best_per_label(self, exp, phase, metric):
        resp = self.get(
            "{url}/results/best/{exp}/{phase}/{metric}".format(
                url=self.url, exp=exp, phase=phase, metric=metric
            )
        )
        return [Label.parse(x) for x in resp['labels']], resp['values'], resp['steps']

    def get_xpctl(self, label):
        resp = _get("{url}/xpctl/{exp}/{sha1}/{name}".format(url=self.url, **label))
        return resp['id']


@export
class RemoteXPCTL(object):
    def __init__(self, host, port, **kwargs):
        self.url = "http://{host}:{port}/hpctl/v1".format(host=host, port=port)

    def put_result(self, label):
        r = requests.post(
            '{url}/xpctl/putresult/{exp}/{sha1}/{name}'.format(
                url=self.url, **label)
        )
        return r.json()['id']

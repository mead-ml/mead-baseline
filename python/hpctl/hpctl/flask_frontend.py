from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip

from multiprocessing import Process
from flask import Flask, request, jsonify, render_template
from hpctl.utils import Label
from hpctl.frontend import Frontend, color


class FlaskFrontend(Frontend):
    def __init__(self, q, results):
        self.results = results
        self.queue = q

    def index(self):
        return render_template('index.html')

    def experiment(self, hash_):
        return jsonify({})

    def experiments(self):
        exps = self.results.get_experiments()
        return jsonify({'experiments': exps})

    def hpctl_config(self, exp):
        return jsonify({})

    def get_config(self, exp, sha1):
        label = Label.parse(sha1)
        res = {
            "sha1": sha1,
            "config": self.results.get_config(label)
        }
        return jsonify(res)

    def best_result(self, exp, label, phase, metric):
        val, idx = self.results.get_best(label, phase, metric)
        res = {
            'label': label,
            'phase': phase,
            'metric': metric,
            'value': val,
            'tick': idx
        }
        return jsonify(res)

    def recent_result(self, exp, label, phase, metric):
        val = self.results.get_rescent(label, phase, metric)
        res = {
            'label': label,
            'phase': phase,
            'metric': metric,
            'value': val
        }
        return jsonify(res)

    def labels(self, exp):
        res = []
        labels = self.results.get_labels(exp)
        for label in labels:
            res.append({"sha1": label.sha1, "human": label.human})
        return jsonify(res)

    def get_label(self, exp, name):
        human = None; sha1 = None
        human, sha1 = self.results.get_label_prefix(name)
        if human is None:
            sha1, human = self.results.get_human_prefix(name)
        return jsonify({"sha1": sha1, "human": human[0]})

    def get_state(self, exp, sha1, name):
        label = Label(exp, sha1, name)
        # Use color to help handle strings.
        state = color(self.results.get_state(label), off=True)
        res = {
            "exp": exp,
            "sha1": sha1,
            "name": name,
            "state": state,
        }
        return jsonify(res)

    def command(self):
        json = request.get_json()
        json['label'] = Label.parse(json['label'])
        self.queue.put(json)
        return jsonify({"success": "success"})

    def kill(self, exp, sha1, name):
        label = Label(exp, sha1, name)
        msg = {
            'command': 'kill',
            'label': label
        }
        self.queue.put(msg)
        return jsonify({"success": "success"})

    def launch(self):
        json = request.get_json()
        json['label'] = Label.parse(json['label'])
        json['command'] = 'launch'
        self.queue.put(json)
        return jsonify({"success": "success"})

    def demo_status(self):
        # THIS FUNCTION IS JUST BECAUSE I DIDN'T WANT A FULL BLOWN JAVASCRIPT APP, TO REMOVE
        labels = self.results.get_labels()
        sha1s = [l.sha1 for l in labels]
        humans = [l.human for l in labels]
        status = [color(self.results.get_state(l), off=True) for l in labels]
        train_stats = [self.results.get_recent(l, 'Train', 'avg_loss') for l in labels]
        train_ticks = [self.results.get_recent(l, 'Train', 'tick') for l in labels]
        dev_stats = [self.results.get_recent(l, 'Valid', 'f1') for l in labels]
        dev_ticks = [self.results.get_recent(l, 'Valid', 'tick') for l in labels]
        res = {
            'status': status,
            'sha1': sha1s,
            'names': humans,
            'exp': [self.exp.experiment_hash] * len(humans),
            'train': train_stats,
            'train_ticks': train_ticks,
            'valid': dev_stats,
            'valid_ticks': dev_ticks
        }
        return jsonify(res)



def init_app(app, fe, base_url='/hpctl/v1'):
    """Bind routes to the functions at runtime so that we can have OOP stuff in the responses."""
    app.route('{}/'.format(base_url), methods={'GET'})(fe.index)
    app.route('{}/config/<exp>'.format(base_url), methods={'GET'})(fe.hpctl_config)
    app.route('{}/config/<exp>/<sha1>'.format(base_url), methods={'GET'})(fe.get_config)
    app.route('{}/result/best/<exp>/<sha1>/<phase>/<metric>'.format(base_url), methods={'GET'})(fe.best_result)
    app.route('{}/result/recent/<exp>/<sha1>/<phase>/<metric>'.format(base_url), methods={'GET'})(fe.recent_result)
    app.route('{}/label/<exp>'.format(base_url), methods={'GET'})(fe.labels)
    app.route('{}/label/<exp>/<name>'.format(base_url), methods={'GET'})(fe.get_label)
    app.route('{}/command'.format(base_url), methods={'POST'})(fe.command)
    app.route('{}/state/<exp>/<sha1>/<name>'.format(base_url), methods={'GET'})(fe.get_state)
    app.route('{}/experiment'.format(base_url), methods={'GET'})(fe.experiments)
    app.route('{}/experiment/<exp>'.format(base_url), methods={'GET'})(fe.experiment)
    app.route('{}/demo'.format(base_url), methods={'GET'})(fe.demo_status)
    app.route('{}/kill/<exp>/<sha1>/<name>'.format(base_url), methods={'GET'})(fe.kill)
    app.route('{}/launch'.format(base_url), methods={'POST'})(fe.launch)


def create_flask(q, results):
    app = Flask(__name__)
    fe = FlaskFrontend(q, results)
    init_app(app, fe)
    p = Process(target=app.run)
    return p

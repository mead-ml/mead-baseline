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

    def best_result(self, exp, sha1, name, phase, metric):
        label = Label(exp, sha1, name)
        val, idx = self.results.get_best(label, phase, metric)
        res = {
            'label': label,
            'phase': phase,
            'metric': metric,
            'value': val,
            'tick': idx
        }
        return jsonify(res)

    def find_best(self, exp, phase, metric):
        label, val, idx = self.results.find_best(exp, phase, metric)
        res = {
            'label': str(label),
            'value': val,
            'step': idx
        }
        return jsonify(res)

    def find_best_per(self, exp, phase, metric):
        labels, vals, idxs = self.results.get_best_per_label(exp, phase, metric)
        res = {
            'labels': [str(x) for x in labels],
            'values': vals,
            'steps': idxs,
        }
        return jsonify(res)

    def recent_result(self, exp, sha1, name, phase, metric):
        label = Label(exp, sha1, name)
        val = self.results.get_recent(label, phase, metric)
        res = {
            'label': str(label),
            'phase': phase,
            'metric': metric,
            'value': val
        }
        return jsonify(res)

    def labels(self, exp):
        res = []
        labels = self.results.get_labels(exp)
        for label in labels:
            res.append({"exp": exp, "sha1": label.sha1, "name": label.name})
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
        names = [l.name for l in labels]
        status = [color(self.results.get_state(l), off=True) for l in labels]
        train_stats = [self.results.get_recent(l, 'Train', 'avg_loss') for l in labels]
        train_ticks = [self.results.get_recent(l, 'Train', 'tick') for l in labels]
        dev_stats = [self.results.get_recent(l, 'Valid', 'f1') for l in labels]
        dev_ticks = [self.results.get_recent(l, 'Valid', 'tick') for l in labels]
        res = {
            'status': status,
            'sha1': sha1s,
            'names': names,
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
    # Get the config for this experiment
    app.route('{}/config/<exp>'.format(base_url), methods={'GET'})(fe.hpctl_config)
    # Get the config based on this sha1
    app.route('{}/config/<exp>/<sha1>'.format(base_url), methods={'GET'})(fe.get_config)
    # Get the best results for this run
    app.route('{}/result/best/<exp>/<sha1>/<name>/<phase>/<metric>'.format(base_url), methods={'GET'})(fe.best_result)
    app.route('{}/result/find/best/<exp>/<phase>/<metric>'.format(base_url), methods={'GET'})(fe.find_best)
    app.route('{}/result/find/best_per/<exp>/<phase>/<metric>'.format(base_url), methods={'GET'})(fe.find_best_per)
    # Get the recent results for this run
    app.route('{}/result/recent/<exp>/<sha1>/<name>/<phase>/<metric>'.format(base_url), methods={'GET'})(fe.recent_result)
    # Get all the labels run for this experiment
    app.route('{}/label/<exp>'.format(base_url), methods={'GET'})(fe.labels)
    # Sha1 look up from name?
    app.route('{}/label/<exp>/<name>'.format(base_url), methods={'GET'})(fe.get_label)
    # General commands
    app.route('{}/command'.format(base_url), methods={'POST'})(fe.command)
    # Get the state of this runn
    app.route('{}/state/<exp>/<sha1>/<name>'.format(base_url), methods={'GET'})(fe.get_state)
    # Get a list of all experiments
    app.route('{}/experiment'.format(base_url), methods={'GET'})(fe.experiments)
    # Get a list of all labels in this experiment?
    app.route('{}/experiment/<exp>'.format(base_url), methods={'GET'})(fe.experiment)
    app.route('{}/demo'.format(base_url), methods={'GET'})(fe.demo_status)
    # Kill this run
    app.route('{}/kill/<exp>/<sha1>/<name>'.format(base_url), methods={'GET'})(fe.kill)
    # Launch this job
    app.route('{}/launch'.format(base_url), methods={'POST'})(fe.launch)


def create_flask(q, results):
    app = Flask(__name__)
    fe = FlaskFrontend(q, results)
    init_app(app, fe)
    p = Process(target=app.run)
    return p

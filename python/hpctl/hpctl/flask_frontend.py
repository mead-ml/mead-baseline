from __future__ import absolute_import, division, print_function, unicode_literals
import six
from six.moves import zip

import math
from multiprocessing import Process
from flask import Flask, request, jsonify, render_template
from hpctl.utils import Label
from hpctl.frontend import Frontend


class FlaskFrontend(Frontend):
    def __init__(self, q, results, xpctl):
        self.results = results
        self.queue = q
        self.xpctl = xpctl

    def index(self):
        return render_template('index.html')

    def experiments(self):
        exps = self.results.get_experiments()
        return jsonify({'experiments': exps})

    def get_labels(self, exp):
        res = []
        labels = self.results.get_labels(exp)
        for label in labels:
            res.append(dict(**label))
        return jsonify(res)

    def exp_config(self, exp):
        exp_config = self.results.get_experiment_config(exp)
        return jsonify(exp_config)

    def get_config(self, exp, sha1, name):
        label = Label(exp, sha1, name)
        res = {
            "config": self.results.get_config(label)
        }
        return jsonify(res)

    def kill(self, exp, sha1, name):
        label = Label(exp, sha1, name)
        msg = {
            'command': 'kill',
            'label': label
        }
        self.queue.put(msg)
        return jsonify({"command": "kill"})

    def launch(self):
        # TODO: Comment on what should be in json to launch
        json = request.get_json()
        json['label'] = Label(json['exp'], json['sha1'], json['name'])
        json['command'] = 'launch'
        self.queue.put(json)
        return jsonify({"command": "launch"})

    def command(self):
        json = request.get_json()
        json['label'] = Label(json['exp'], json['sha1'], json['name'])
        self.queue.put(json)
        return jsonify({"command": "success"})

    def add_experiment(self):
        json = request.get_json()
        self.results.add_experiment(json)
        return jsonify({"command": "add"})

    def put_result(self, exp, sha1, name):
        label = Label(exp, sha1, name)
        if self.xpctl is None:
            return jsonify({"command": "putresults", "status": "failed", "id": "FAILED"})
        if not self.results.get_xpctl(label):
            id_ = self.xpctl.put_result(label)
            self.results.set_xpctl(label, id_)
            id_ = str(id_)
        else:
            id_ = self.results.get_xpctl(label)
        return jsonify({"command": "putresults", "status": "success", "id": id_})

    def get_xpctl(self, exp, sha1, name):
        label = Label(exp, sha1, name)
        id_ = self.results.get_xpctl(label)
        return jsonify(dict(id=id_, **label))

    def get_state(self, exp, sha1, name):
        label = Label(exp, sha1, name)
        state = self.results.get_state(label)
        state = str(state).decode('utf-8') if six.PY2 else str(state)
        res = dict(state=state, **label)
        return jsonify(res)

    def recent_result(self, exp, sha1, name, phase, metric):
        label = Label(exp, sha1, name)
        val = self.results.get_recent(label, phase, metric)
        res = dict(phase=phase, metric=metric, value=val, **label)
        return jsonify(res)

    def best_result(self, exp, sha1, name, phase, metric):
        label = Label(exp, sha1, name)
        val, idx = self.results.get_best(label, phase, metric)
        res = dict(phase=phase, metric=metric, value=val, tick=idx, **label)
        return jsonify(res)

    def find_best_across(self, exp, phase, metric):
        label, val, idx = self.results.find_best(exp, phase, metric)
        res = dict(phase=phase, metric=metric, value=val, tick=idx, **label)
        return jsonify(res)

    def find_best_within(self, exp, phase, metric):
        labels, vals, idxs = self.results.get_best_per_label(exp, phase, metric)
        res = {
            'exps': [l.exp for l in labels],
            'sha1s': [l.sha1 for l in labels],
            'names': [l.name for l in labels],
            'values': vals,
            'steps': idxs,
        }
        return jsonify(res)

    def demo_data(self, exp):
        # THIS FUNCTION IS JUST BECAUSE I DIDN'T WANT A FULL BLOWN JAVASCRIPT APP, TO REMOVE
        # Nan is not part of json so we need  to filter it out.
        labels = self.results.get_labels(exp)
        sha1s = [l.sha1 for l in labels]
        names = [l.name for l in labels]
        status = [self.results.get_state(l) for l in labels]
        status = [str(s).decode('utf-8') if six.PY2 else str(s) for s in status]
        train_stats = [self.results.get_recent(l, 'Train', 'avg_loss') for l in labels]
        train_stats = [x if not math.isnan(x) else 0.0 for x in train_stats]
        train_ticks = [self.results.get_recent(l, 'Train', 'tick') for l in labels]
        dev_stats = [self.results.get_recent(l, 'Valid', 'f1') for l in labels]
        dev_stats = [x if not math.isnan(x) else 0.0 for x in dev_stats]
        dev_ticks = [self.results.get_recent(l, 'Valid', 'tick') for l in labels]
        xpctls = [self.results.get_xpctl(l) for l in labels]
        res = {
            'status': status,
            'sha1': sha1s,
            'names': names,
            'exp': [exp] * len(names),
            'train': train_stats,
            'train_ticks': train_ticks,
            'valid': dev_stats,
            'valid_ticks': dev_ticks,
            'xpctls': xpctls
        }
        return jsonify(res)

    def demo_page(self, exp):
        return render_template('demo.html', exp_hash=exp)

    def get_label(self, name):
        human, label = self.results.get_label_prefix(name)
        if human is None:
            return jsonify({"exp": None, "sha1": None, "name": None})
        return jsonify(dict(**label))

    def get_metrics(self, exp, sha1, name, phase):
        label = Label(exp, sha1, name)
        metrics = self.results.get_metrics(label, phase)
        return jsonify(dict(phase=phase, metrics=metrics, **label))


def init_app(app, fe, base_url='/hpctl/v1'):
    """Bind routes to the functions at runtime so that we can have OOP stuff in the responses."""
    app.route('/'.format(base_url), methods={'GET'})(fe.index)
    app.route('{}/'.format(base_url), methods={'GET'})(fe.index)
    ## Exploratory
    # Get the list of all experiments
    app.route('{}/experiments'.format(base_url), methods={'GET'})(fe.experiments)
    # Get the list of all runs within an experiment
    app.route('{}/labels/<exp>'.format(base_url), methods={'GET'})(fe.get_labels)

    ## Configs
    # Get the config for this experiment
    app.route('{}/config/<exp>'.format(base_url), methods={'GET'})(fe.exp_config)
    # Get the config for this job
    app.route('{}/config/<exp>/<sha1>/<name>'.format(base_url), methods={'GET'})(fe.get_config)

    ## Control
    # Kill this job
    app.route('{}/kill/<exp>/<sha1>/<name>'.format(base_url), methods={'POST'})(fe.kill)
    # Launch a job
    app.route('{}/launch'.format(base_url), methods={'POST'})(fe.launch)
    # General commands (Currently unused)
    app.route('{}/command'.format(base_url), methods={'POST'})(fe.command)
    # Add an experiment config to look at later
    app.route('{}/experiment/add'.format(base_url), methods={'POST'})(fe.add_experiment)
    # Store the results of a job with xpctl
    app.route('{}/xpctl/putresult/<exp>/<sha1>/<name>'.format(base_url), methods={'POST'})(fe.put_result)

    ## Information
    # Get the state of this run
    app.route('{}/state/<exp>/<sha1>/<name>'.format(base_url), methods={'GET'})(fe.get_state)
    # Get the most recent data for this run
    app.route('{}/result/recent/<exp>/<sha1>/<name>/<phase>/<metric>'.format(base_url), methods={'GET'})(fe.recent_result)
    # Get the best result this run has had
    app.route('{}/result/best/<exp>/<sha1>/<name>/<phase>/<metric>'.format(base_url), methods={'GET'})(fe.best_result)
    # Get the best results any run has had (compares across runs)
    app.route('{}/result/best/<exp>/<phase>/<metric>'.format(base_url), methods={'GET'})(fe.find_best_across)
    # Get the best results for each run (compares withing runs)
    app.route('{}/results/best/<exp>/<phase>/<metric>'.format(base_url), methods={'GET'})(fe.find_best_within)
    # Check if this has be logged to xpctl
    app.route('{}/xpctl/<exp>/<sha1>/<name>'.format(base_url), methods={'GET'})(fe.get_xpctl)
    # Get the label based on the name
    app.route('{}/label/<name>'.format(base_url), methods={'GET'})(fe.get_label)
    # Get all available metrics
    app.route('{}/metrics/<exp>/<sha1>/<name>/<phase>'.format(base_url), methods={'GET'})(fe.get_metrics)

    # Simple display
    app.route('{}/demo_data/<exp>'.format(base_url), methods={'GET'})(fe.demo_data)
    app.route('{}/demo_page/<exp>'.format(base_url), methods={'GET'})(fe.demo_page)


def create_flask(q, results, xpctl):
    app = Flask(__name__)
    fe = FlaskFrontend(q, results, xpctl)
    init_app(app, fe)
    p = Process(target=app.run)
    return p

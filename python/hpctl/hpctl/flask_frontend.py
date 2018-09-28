from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import zip

import math
from multiprocessing import Process
from flask import Flask, request, jsonify, render_template
from hpctl.utils import Label
from hpctl.frontend import Frontend, color


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
        json['label'] = Label.parse(json['label'])
        json['command'] = 'launch'
        self.queue.put(json)
        return jsonify({"command": "launch"})

    def command(self):
        json = request.get_json()
        json['label'] = Label.parse(json['label'])
        self.queue.put(json)
        return jsonify({"command": "success"})

    def add_experiment(self):
        json = request.get_json()
        self.results.add_experiment(json)
        return jsonify({"command": "add"})

    def put_result(self, exp, sha1, name):
        label = Label(exp, sha1, name)
        self.xpctl.put_result(label)
        return jsonify({"command": "putresults", "status": "success"})

    def get_state(self, exp, sha1, name):
        label = Label(exp, sha1, name)
        # Using color to help handle strings.
        state = color(self.results.get_state(label), off=True)
        res = {
            "exp": exp,
            "sha1": sha1,
            "name": name,
            "state": state,
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

    def best_result(self, exp, sha1, name, phase, metric):
        label = Label(exp, sha1, name)
        val, idx = self.results.get_best(label, phase, metric)
        res = {
            'label': str(label),
            'phase': phase,
            'metric': metric,
            'value': val,
            'tick': idx
        }
        return jsonify(res)

    def find_best_across(self, exp, phase, metric):
        label, val, idx = self.results.find_best(exp, phase, metric)
        res = {
            'label': str(label),
            'value': val,
            'step': idx
        }
        return jsonify(res)

    def find_best_within(self, exp, phase, metric):
        labels, vals, idxs = self.results.get_best_per_label(exp, phase, metric)
        res = {
            'labels': [str(x) for x in labels],
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
        status = [color(self.results.get_state(l), off=True) for l in labels]
        train_stats = [self.results.get_recent(l, 'Train', 'avg_loss') for l in labels]
        train_stats = [x if not math.isnan(x) else 0.0 for x in train_stats]
        train_ticks = [self.results.get_recent(l, 'Train', 'tick') for l in labels]
        dev_stats = [self.results.get_recent(l, 'Valid', 'f1') for l in labels]
        dev_stats = [x if not math.isnan(x) else 0.0 for x in dev_stats]
        dev_ticks = [self.results.get_recent(l, 'Valid', 'tick') for l in labels]
        res = {
            'status': status,
            'sha1': sha1s,
            'names': names,
            'exp': [exp] * len(names),
            'train': train_stats,
            'train_ticks': train_ticks,
            'valid': dev_stats,
            'valid_ticks': dev_ticks
        }
        return jsonify(res)

    def demo_page(self, exp):
        return render_template('demo.html', exp_hash=exp)

    # def get_label(self, exp, name):
    #     human = None; sha1 = None
    #     human, sha1 = self.results.get_label_prefix(name)
    #     if human is None:
    #         sha1, human = self.results.get_human_prefix(name)
    #     return jsonify({"sha1": sha1, "human": human[0]})


def init_app(app, fe, base_url='/hpctl/v1'):
    """Bind routes to the functions at runtime so that we can have OOP stuff in the responses."""
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



    # # Get all the labels run for this experiment
    # app.route('{}/label/<exp>'.format(base_url), methods={'GET'})(fe.labels)
    # # Sha1 look up from name?
    # app.route('{}/label/<exp>/<name>'.format(base_url), methods={'GET'})(fe.get_label)
    # # Get a list of all experiments
    # Simple display
    app.route('{}/demo_data/<exp>'.format(base_url), methods={'GET'})(fe.demo_data)
    app.route('{}/demo_page/<exp>'.format(base_url), methods={'GET'})(fe.demo_page)


def create_flask(q, results, xpctl):
    app = Flask(__name__)
    fe = FlaskFrontend(q, results, xpctl)
    init_app(app, fe)
    p = Process(target=app.run)
    return p

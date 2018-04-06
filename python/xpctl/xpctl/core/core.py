import click
from xpctl.helpers import *
import pandas as pd
import os
import subprocess
import pymongo
import shutil
from bson.objectid import ObjectId


def connect(host, port, user, passw):
    client = None
    if user and passw:
        uri = "mongodb://{}:{}@{}:{}/test".format(user, passw, host, port)
        client = pymongo.MongoClient(uri)
    else:
        client = pymongo.MongoClient(host, port)
    if client is None:
        print("can not connect to mongo at host: [{}], port [{}], username: [{}], password: [{}]".format(host, port,
                                                                                                         user, passw))
        return None
    try:
        dbnames = client.database_names()
    except pymongo.errors.ServerSelectionTimeoutError:
        print("can not get database from mongo at host: {}, port {}, connection timed out".format(host, port))
        return None
    if "reporting_db" not in dbnames:
        print("no database for results found")
        return None
    return client.reporting_db


dbhost = None
dbport = None
db = None

events = {
    "train": "train_events",
    "test": "test_events",
    "valid": "valid_events",
    "dev": "valid_events",
}


def dbsetup(tname):
    if db is None:
        click.echo("set up db connection using setupdb command")
        return False
    elif tname not in db.collection_names():
        click.echo("no results for the specified task {}, use another task".format(tname))
        return False
    else:
        return True


def cli_int(dbhost, dbport, dbuser, dbpass):
    global db
    db = connect(dbhost, dbport, dbuser, dbpass)
    return db


def get_modelloc_int(task, id):
    if not dbsetup(task):
        return None
    coll = db[task]
    query = {'_id': ObjectId(id)}
    projection = {'checkpoint': 1}
    results = [x.get('checkpoint', None) for x in list(coll.find(query, projection))]
    results = [x for x in results if x]
    if not results:
        return None
    return results[0]


def get_metrics(list, event_type):
    keys = []
    for x in list:
        if x[event_type]:
            for k in x[event_type][0].keys():
                keys.append(k)
    keys = set(keys)
    if 'tick_type' in keys: keys.remove("tick_type")
    if 'tick' in keys: keys.remove("tick")
    if 'phase' in keys: keys.remove("phase")
    return keys


def generate_data_frame(coll, metrics, query, projection, event_type):
    results = list(coll.find(query, projection))
    if not results:
        return

    ms = list(set(metrics)) if metrics else list(get_metrics(results, event_type))
    presults = []
    for result in results:  # different experiments
        for index in range(len(result[event_type])):  # train_event epoch 0,
            # train_event epoch 1 etc, for event_type = test_event, there is only one event
            data = []
            for metric in ms:
                data.append(result[event_type][index][metric])
            presults.append(
                [result['_id'], result['username'], result['label'], result['config']['dataset'], result['sha1'],
                 result['date']] + data)
    return pd.DataFrame(presults, columns=['id', 'username', 'label', 'dataset', 'sha1', 'date'] + ms)


def update_query(q, uname, dataset):
    query = q
    if uname:
        query.update({"username": {"$in": list(uname)}})
    if dataset:
        query.update({"config.dataset": dataset})
    return query


def update_projection(event_type):
    projection = {"_id": 1, "sha1": 1, "label": 1, "username": 1, "config.dataset": 1, "date": 1}
    projection.update({event_type: 1})
    return projection


def bestn_results(uname, metric, dataset, tname, numresults, event_type, ascending):
    if not dbsetup(tname):
        return None
    else:
        metrics = [metric]
        click.echo("using metric: {}".format(metrics))
        coll = db[tname]
        query = update_query({}, uname, dataset)
        projection = update_projection(event_type)
        resultdframe = generate_data_frame(coll, metrics, query, projection, event_type)
        if not resultdframe.empty:
            resultdframe = resultdframe.sort_values(metrics, ascending=[ascending])[
                           :min(int(numresults), resultdframe.shape[0])]
            return resultdframe
        else:
            return None


def results_int(user, metric, sort, dataset, task, event_type):
    event_type = event_type.lower()

    if not dbsetup(task):
        return None
    elif event_type not in events:
        click.echo("we do not have results for the event type: {}".format(event_type))
    else:
        metrics = list(metric)
        coll = db[task]
        query = update_query({}, user, dataset)
        projection = update_projection(event_type=events[event_type])
        resultdframe = generate_data_frame(coll, metrics, query, projection, event_type=events[event_type])
        if len(metric) == 1:
            metric = metric[0]
            if metric == "avg_loss" or metric == "perplexity":
                resultdframe = resultdframe.sort_values(metric, ascending=True)
            else:
                resultdframe = resultdframe.sort_values(metric, ascending=False)
        if sort:
            if sort == "avg_loss" or sort == "perplexity":
                resultdframe = resultdframe.sort_values(sort, ascending=True)
            else:
                resultdframe = resultdframe.sort_values(sort, ascending=False)
        if resultdframe is None:
            return None
        elif not resultdframe.empty:
            return resultdframe
        else:
            return None


def config2json_int(task, sha):
    """Exports the config file for an experiment as a json file. Arguments: taskname,
    experiment sha1, output file path"""
    if not dbsetup(task):
        return None
    else:
        coll = db[task]
        j = coll.find_one({"sha1": sha}, {"config": 1})["config"]
        if not j:
            return None
        else:
            return j


def best_int(user, metric, dataset, n, task, event_type):
    """Shows the best F1 score for event_type(tran/valid/test) on a particular task (classify/ tagger) on
    a particular dataset (SST2, wnut) using a particular metric. Default behavior: The best result for
    **All** users available for the task. Optionally supply number of results (n-best), user(s) and metric(only ONE)"""
    event_type = event_type.lower()

    if event_type not in events:
        return None
    else:
        if metric == "avg_loss" or metric == "perplexity":
            return bestn_results(user, metric, dataset, task, n, events[event_type], ascending=True)
        else:
            return bestn_results(user, metric, dataset, task, n, events[event_type], ascending=False)


def task_summary(task, dataset, metric):
    if not dbsetup(task):
        return None
    else:
        metrics = [metric]
        coll = db[task]
        query = update_query({}, [], dataset)
        projection = update_projection(event_type=events["test"])
        resultdframe = generate_data_frame(coll, metrics, query, projection, event_type=events["test"])
        if not resultdframe.empty:
            datasets = resultdframe.dataset.unique()
            if dataset not in datasets:
                click.echo("no result found for the requested dataset: {}".format(dataset))
                return
            dsr = resultdframe[resultdframe.dataset == dataset].sort_values(metric, ascending=False)
            result = dsr[metric].iloc[0]
            user = dsr.username.iloc[0]
            sha1 = dsr.sha1.iloc[0]
            date = dsr.date.iloc[0]
            summary = "For dataset {}, the best {} is {:0.3f} reported by {} on {}. " \
                      "The sha1 for the config file is {}.".format(dataset, metric, result, user, date, sha1)
            return summary
        else:
            return None


def generate_info(coll):
    """we will show what datasets are available for this task, what are the metrics and which username and hostnames
    have participated in this."""
    event_types = [events["train"], events["test"], events["valid"]]

    q = update_query({}, None, None)
    p = {'config.dataset': 1}
    datasets = list(set([x['config']['dataset'] for x in list(coll.find(q, p))]))
    store = []  #

    for dataset in datasets:
        q = update_query({}, None, dataset)
        for event_type in event_types:
            p = update_projection(event_type)
            results = list(coll.find(q, p))
            metrics = get_metrics(results, event_type)
            for result in results:  # different experiments
                store.append([result['username'], result['config']['dataset'], event_type, ",".join(metrics)])

    df = pd.DataFrame(store, columns=['user', 'dataset', 'event_type', 'metrics'])
    return df.groupby(['user', 'dataset', 'event_type', 'metrics']).size().reset_index() \
        .rename(columns={0: 'num_experiments'})


def get_git_revision_hash(baselinepath):
    return subprocess.check_output(['git', '--git-dir={}/.git'.format(baselinepath), 'rev-parse', 'HEAD']).strip()


def get_baseline_loc():  # assumes baseline is in pythonpath
    return [x for x in os.environ['PYTHONPATH'].split(":") if "baseline" in x][0][:-7]


def get_baseline_sha1():
    return get_git_revision_hash(get_baseline_loc()).strip().decode("utf-8")


def storemodel(cbase, configsha1, cstore):
    mdir, mbase = os.path.split(cbase)
    mdir = mdir if mdir else "."
    if not os.path.exists(mdir):
        click.echo("no directory found for the model location: [{}], aborting command".format(mdir))
        return None

    mfiles = ["{}/{}".format(mdir, x) for x in os.listdir(mdir) if x.startswith(mbase + "-") or
              x.startswith(mbase + ".")]
    if not mfiles:
        click.echo("no model files found with the model base [{}] at the model location [{}], aborting command"
                   .format(mbase, mdir))
        return None
    modellocbase = "{}/{}".format(cstore, configsha1)
    if not os.path.exists(modellocbase):
        os.makedirs(modellocbase)
    dirs = [int(x[:-4]) for x in os.listdir(modellocbase) if x.endswith(".zip") and x[:-4].isdigit()]
    # we expect dirs in numbers.
    newdir = "1" if not dirs else str(max(dirs) + 1)
    modelloc = "{}/{}".format(modellocbase, newdir)
    os.makedirs(modelloc)
    for mfile in mfiles:
        shutil.copy(mfile, modelloc)
        click.echo("writing model file: [{}] to store: [{}]".format(mfile, modelloc))
    click.echo("zipping model files")
    shutil.make_archive(base_name=modelloc,
                        format='zip',
                        root_dir=modellocbase,
                        base_dir=newdir)
    shutil.rmtree(modelloc)
    click.echo("zipped file written, model directory removed")
    return modelloc + ".zip"

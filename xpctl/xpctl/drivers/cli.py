from click_shell import shell
from xpctl.core import *
import pandas as pd
import datetime
import socket
import hashlib
import os
import getpass
from bson.objectid import ObjectId
import sys
import json

pd.set_option('display.expand_frame_repr', False)

dbhost = None
dbport = None
dbuser = None
dbpass = None
db = None


events = {
    "train": "train_events", "Train": "train_events",
    "test": "test_events", "Test": "test_events",
    "valid": "valid_events", "Valid": "valid_events",
    "dev": "valid_events", "Dev": "valid_events"
}


# set up env
def read_cred(configjson):
    j = json.load(open(configjson))
    try:
        return(j.get('dbhost', None), j.get('dbport',None), j.get('user',None), j.get('passwd',None))
    except IOError:
        return (None, None, None, None)

@shell(prompt="xpctl > ", intro="Starting xpctl...")
@click.option('--host', help="mongo host")
@click.option('--port', help="mongo port", default=27017)
@click.option('--user', help="mongo username")
@click.option('--password', help="mongo password")
@click.option('--config', help="mongo creds", default="~/xpctlcred.json")
def cli(host, port, user, password, config):
    global dbhost
    global dbport
    global dbuser
    global dbpass
    global db
    host_c,port_c,user_c,passw_c = (None, None, None, None)
    if os.path.exists(os.path.expanduser(config)):
        host_c,port_c,user_c,passw_c = read_cred(os.path.expanduser(config))

    dbhost = host if host else host_c
    dbport = port if port else port_c 
    dbuser = user if user else user_c
    dbpass = password if password else passw_c 
    creds = {'mongo host':(dbhost, 'host'), 'mongo port':(dbport, 'port'), 'username':(dbuser, 'user'), 'password': (dbpass, 'password')}
    for var in creds:
        if creds[var][0] is None:
            click.echo("Value for [{}] is None, provide a proper value using --{} option".format(var,creds[var][1])) 
            sys.exit(1)

    db = cli_int(dbhost, dbport, dbuser, dbpass)
    if db is not None:
        click.echo("db connection successful with [host]: {}, [port]: {}".format(dbhost, dbport))
    else:
        click.echo("db connection unsuccessful, aborting")
        sys.exit(1)


@cli.command()
def vars():
    """Prints the value of system variables dbhost and dbport"""
    click.echo("dbhost: {}, dbport: {}".format(dbhost, dbport))


@cli.command()
@click.argument('task')
@click.argument('id')
def getmodelloc(task, id):
    """get the model location for a particluar task and record id"""
    result = get_modelloc_int(task,id)  
    if result is None:
        click.echo("no results found")
        return
    click.echo("model loc is {}".format(result))
    return 


@cli.command()
@click.option('--user', multiple=True, help="list of users (testuser, root), [multiple]: --user a --user b")
@click.option('--metric', multiple=True, help="list of metrics (prec, recall, f1, accuracy),[multiple]: --metric f1 "
                                              "--metric acc")
@click.option('--sort', help="specify one metric to sort the results")
@click.argument('task')
@click.argument('event_type')
@click.argument('dataset')
def results(user, metric, sort, dataset, task, event_type):
    """Shows the results for event_type(tran/valid/test) on a particular task (classify/ tagger) with a particular
    dataset (SST2, wnut). Default behavior: **All** users, and metrics. Optionally supply user(s), metric(s), a sort
    metric. use --metric f1 --metric acc for multiple metrics. use one metric for the sort.
    """
    resultdframe = results_int(user, metric, sort, dataset, task, event_type)
    if resultdframe is not None:
            click.echo(resultdframe)
            resultdframe.desc = "{} results for task: {}".format(event_type, task)
    else:
            click.echo("no result found for this query")


@cli.command()
@click.option('--user', multiple=True, help="list of users (testuser, root), [multiple]: --user a --user b")
@click.option('--n', default=1, help="N best results")
@click.argument('task')
@click.argument('event_type')
@click.argument('dataset')
@click.argument('metric')
def best(user, metric, dataset, n, task, event_type):
    """Shows the best F1 score for event_type(tran/valid/test) on a particular task (classify/ tagger) on
    a particular dataset (SST2, wnut) using a particular metric. Default behavior: The best result for
    **All** users available for the task. Optionally supply number of results (n-best), user(s) and metric(only ONE)"""

    if event_type not in events:
        click.echo("we do not have results for the event type: {}".format(event_type))
        return
    resultdframe = best_int(user, metric, dataset, n, task, event_type)
    if resultdframe is not None:
        click.echo("total {} results found, showing best {} results".format(resultdframe.shape[0], n))
        click.echo(resultdframe)
    else:
        click.echo("no result found for this query")


# summarize results
@cli.command()
@click.option('--task')
def lbsummary(task):
    """Provides a summary of the leaderboard. Options: taskname. If you provide a taskname, it will show
    all users, datasets and metrics for that task. This is helpful because we often forget what metrics or datasets were
    used for a task, which are the necessary parameters for the commands `results` and `best` and `tasksummary`. Shows
    the summary for all available tasks if no option is specified."""
    db = cli_int(dbhost, dbport, dbuser, dbpass)
    if db is None:
        click.echo("can not connect to database")
        return
    if task:
        click.echo("Task: [{}]".format(task))
        click.echo("---------------------------------------------------------------------------------------------")
        click.echo(generate_info(db[task]))
    else:
        tasks = db.collection_names()
        if "system.indexes" in tasks:
            tasks.remove("system.indexes")
        click.echo("There are {} tasks: {}".format(len(tasks), tasks))
        for task in tasks:
            click.echo("---------------------------------------------------------------------------------------------")
            click.echo("Task: [{}]".format(task))
            click.echo("---------------------------------------------------------------------------------------------")
            click.echo(generate_info(db[task]))


@cli.command()
@click.argument('task')
@click.argument('dataset')
@click.argument('metric')
def tasksummary(task, dataset, metric):
    """Provides a natural language summary for a task. This is almost equivalent to the `best` command."""
    tsummary = task_summary(task, dataset, metric)
    if tsummary is None:
        click.echo("can't produce summary for the requested task {}".format(task))
        return
    click.echo(tsummary)


# Edit database
@cli.command()
@click.argument('task')
@click.argument('id')
@click.argument('label')
def updatelabel(id, label, task):
    """Update the _label_ for an experiment (identified by its id) for a task"""
    db = cli_int(dbhost, dbport, dbuser, dbpass)
    if db is None:
        click.echo("can not connect to database")
        return
    else:
        coll = db[task]
        prevlabel = coll.find_one({'_id': ObjectId(id)}, {'label': 1})["label"]
        click.echo("[previous label for the experiment]: {} ".format(prevlabel))
        coll.update({'_id': ObjectId(id)}, {'$set': {'label': label}}, upsert=False)
        changedlabel = coll.find_one({'_id': ObjectId(id)}, {'label': 1})["label"]
        click.echo("[updated label for the experiment]: {} ".format(changedlabel))


@cli.command()
@click.argument('task')
@click.argument('id')
def delete(id, task):
    """delete a record from database with the given object id. also delete the associated model file from the checkpoint."""
    db = cli_int(dbhost, dbport, dbuser, dbpass)
    if db is None:
        click.echo("can not connect to database")
        return
    else:
        coll = db[task]
        prev = coll.find_one({'_id': ObjectId(id)}, {'label': 1})
        if prev is None:
            click.echo("The record {} doesn't exist in the database".format(id))
            return 
        click.echo("You are going to delete the record {} from {} database. We will also delete the model file if it exists.".format(prev['label'], task))
        if click.confirm('Do you want to continue?'):
            modelloc = getmodelloc_int(task,id)
            if modelloc is not None:
                if os.path.exists(modelloc):
                    os.remove(modelloc)
            else:
                click.echo("No model stored for this record. Only purging the database.")  
            coll.remove({'_id': ObjectId(id)})  
            assert(coll.find_one({'_id': ObjectId(id)}) is None)
            click.echo("record {} deleted successfully from database {}".format(id, task))
        else:
            click.echo("no record deleted")
        return   


# Put results in database
@cli.command()
@click.option("--user", default=getpass.getuser(), help="username")
@click.option("--cbase", help="path to the base structure for the model checkpoint files:"
                              "such as ../tagger/tagger-model-tf-11967 or /home/ds/tagger/tagger-model-tf-11967")
@click.option("--cstore", default="/data/model-checkpoints", help="location of the model checkpoint store")
@click.argument('task')
@click.argument('config')
@click.argument('log')
@click.argument('label')
def putresult(user, log, task, config, label, cbase, cstore):
    """Puts the results of an experiment on the database. Arguments:  task name, location of the config file for
    the experiment, the log file storing the results for the experiment (typically <taskname>/reporting.log)
    and a short description of the experiment (label). Gets the username from system (can provide as an option). Also
    provide the model location produced by the config optionally"""
    db = cli_int(dbhost, dbport, dbuser, dbpass)
    if db is None:
        click.echo("can not connect to database")
        return
    logf = log.format(task)
    if not os.path.exists(logf):
        click.echo("the log file at {} doesn't exist, provide a valid location".format(logf))
        return
    if not os.path.exists(config):
        click.echo("the config file at {} doesn't exist, provide a valid location".format(config))
        return

    configf = config
    config = read_config(config)
    now = datetime.datetime.utcnow()
    events = log2json(logf)
    train_events = list(filter(lambda x: x['phase'] == 'Train', events))
    valid_events = list(filter(lambda x: x['phase'] == 'Valid', events))
    test_events = list(filter(lambda x: x['phase'] == 'Test', events))
    hostname = socket.gethostname()

    post = {
        "config": config,
        "train_events": train_events,
        "valid_events": valid_events,
        "test_events": test_events,
        "username": user,
        "hostname": hostname,
        "date": now,
        "label": label,
        "sha1": hashlib.sha1(open(configf, "rb").read()).hexdigest(),  # uniquely identifies an
        # experiment by taking sha1 of a cofig file.
        "baslinegitsha1": get_baseline_sha1()
    }

    if cbase:
        modelloc = storemodel(cbase=cbase, configsha1=hashlib.sha1(open(configf, "rb").read()).hexdigest(),
                              cstore=cstore)
        if modelloc is not None:
            post.update({"checkpoint": "{}:{}".format(hostname, os.path.abspath(modelloc))})
        else:
            click.echo("model could not be stored, see previous errors")

    if task in db.collection_names():
        click.echo("updating results for existing task [{}] in host [{}]".format(task, dbhost))
    else:
        click.echo("creating new task [{}] in host [{}]".format(task, dbhost))
    coll = db[task]
    insertoneresult = coll.insert_one(post)
    click.echo("results updated, the new results are stored with the record id: {}".format(insertoneresult.inserted_id))


@cli.command()
@click.option("--cstore", default="/data/model-checkpoints", help="location of the model checkpoint store")
@click.argument('task')
@click.argument('id')
@click.argument('cbase')
def putmodel(task, id, cbase, cstore):
    """Puts the model from an experiment in the model store and updates the database with the location.
    Arguments:  task name, id of the record, and the path to the base structure for
    the model checkpoint files such as ../tagger/tagger-model-tf-11967 or /home/ds/tagger/tagger-model-tf-11967"""
    db = cli_int(dbhost, dbport, dbuser, dbpass)
    if db is None:
        click.echo("can not connect to database")
        return
    coll = db[task]
    query = {'_id':ObjectId(id)} 
    projection = {'sha1':1}
    results = list(coll.find(query,projection))
    if not results:
        click.echo("no sha1 for the given id found, returning.")
        return
    sha1 = results[0]['sha1']
    modelloc = storemodel(cbase=cbase, configsha1=sha1, cstore=cstore)
    if modelloc is not None: #update the collection with sha1
        coll = db[task]
        coll.update_one({'_id': ObjectId(id)}, {'$set': {'checkpoint': modelloc}}, upsert=False)
        click.echo("database updated with modelloc")
    else:
        click.echo("model could not be stored, see previous errors")



# writers
@cli.command()
@click.argument("task")
@click.argument("sha")
@click.argument("filename")
def config2json(task, sha, filename):
    """Exports the config file for an experiment as a json file. Arguments: taskname,
    experiment sha1, output file path"""
    j = config2json_int(task, sha)
    if j is None:
        click.echo("can not find config sha1: {}".format(sha))
    else:
        json.dump(j, open(filename, "w"), indent=True)


@cli.command()
@click.argument("task")
@click.argument("sha1")
@click.argument("sha2")
def configdiff(task, sha1, sha2):
    """Shows the difference between two json files for a task, diff of sha2 wrt sha1"""
    db = cli_int(dbhost, dbport, dbuser, dbpass)
    if db is None:
        click.echo("can not connect to database")
        return
    else:
        coll = db[task]
        j1 = coll.find_one({"sha1": sha1}, {"config": 1})["config"]
        j2 =  coll.find_one({"sha1": sha2}, {"config": 1})["config"]
        if not j1:
            click.echo("can not find config sha1: {}".format(sha1))
        elif not j2:
            click.echo("can not find config sha2: {}".format(sha2))
        else:
            from jsondiff import diff
            click.echo("diff of sha2 wrt sha1")
            click.echo(diff(j1, j2))


if __name__ == "__main__":
    cli()

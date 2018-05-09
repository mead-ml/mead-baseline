import json
import os
import sys
import pandas as pd
import getpass
from click_shell import shell
import click
from xpctl.core import ExperimentRepo
from xpctl.helpers import *
pd.set_option('display.expand_frame_repr', False)


class RepoManager(object):

    central_repo = None
    dbhost = None
    dbport = None
    dbuser = None
    dbpass = None

    @staticmethod
    def get():
        if RepoManager.central_repo is None:
            RepoManager.central_repo = ExperimentRepo.create_mongo_repo(RepoManager.dbhost, RepoManager.dbport,
                                                                        RepoManager.dbuser, RepoManager.dbpass)

        if RepoManager.central_repo is not None:
            click.echo("db connection successful with [host]: {}, [port]: {}".format(RepoManager.dbhost,
                                                                                     RepoManager.dbport))
            return RepoManager.central_repo
        click.echo("db connection unsuccessful, aborting")
        sys.exit(1)


EVENT_TYPES = {
    "train": "train_events", "Train": "train_events",
    "test": "test_events", "Test": "test_events",
    "valid": "valid_events", "Valid": "valid_events",
    "dev": "valid_events", "Dev": "valid_events"
}


# set up env
def read_cred(config_json):
    with open(config_json) as f:
        j = json.load(f)
    try:
        return j.get('dbhost', None), j.get('dbport', None), j.get('user', None), j.get('passwd', None)
    except IOError:
        return None, None, None, None


@shell(prompt="xpctl > ", intro="Starting xpctl...")
@click.option('--host', help="mongo host")
@click.option('--port', help="mongo port", default=27017)
@click.option('--user', help="mongo username")
@click.option('--password', help="mongo password")
@click.option('--config', help="mongo creds", default="~/xpctlcred.json")
def cli(host, port, user, password, config):

    host_c, port_c, user_c, passw_c = (None, None, None, None)
    if os.path.exists(os.path.expanduser(config)):
        host_c, port_c, user_c, passw_c = read_cred(os.path.expanduser(config))

    if host is None and host_c is None:
        host = "localhost"
    elif host is None:
        host = host_c

    if port is None and port_c is None:
        port = 27017
    elif port is None:
        port = port_c

    RepoManager.dbhost = host
    RepoManager.dbport = port
    RepoManager.dbuser = user if user else user_c
    RepoManager.dbpass = password if password else passw_c


@cli.command()
def vars():
    """Prints the value of system variables dbhost and dbport"""
    click.echo("dbhost: {}, dbport: {}".format(RepoManager.dbhost, RepoManager.dbport))


@cli.command()
@click.argument('task')
@click.argument('id')
def getmodelloc(task, id):
    """get the model location for a particular task and record id"""
    if not RepoManager.get().has_task(task):
        click.echo("no results for the specified task {}, use another task".format(task))
        return

    result = RepoManager.get().get_model_location(id, task)

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
    """
    Shows the results for event_type(tran/valid/test) on a particular task
    (classify/ tagger) with a particular dataset (SST2, wnut).

    Default behavior: **All** users, and metrics. Optionally supply user(s),
    metric(s), a sort metric. use --metric f1 --metric acc for multiple metrics.
    use one metric for the sort.
    """
    event_type = event_type.lower()

    if not RepoManager.get().has_task(task):
        click.echo("no results for the specified task {}, use another task".format(task))
        return

    event_type = EVENT_TYPES.get(event_type, None)
    if event_type is None:
        click.echo("we do not have results for the event type: {}".format(event_type))
        return

    result_frame = RepoManager.get().get_results(user, metric, sort, dataset, task, event_type)

    if result_frame is not None:
        click.echo(result_frame)
        result_frame.desc = "{} results for task: {}".format(event_type, task)
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
    """
    Shows the best F1 score for event_type(tran/valid/test) on a
    particular task (classify/ tagger) on a particular dataset (SST2, wnut)
    using a particular metric. Default behavior: The best result for
    **All** users available for the task. Optionally supply number of results
    (n-best), user(s) and metric(only ONE)
    """

    event_type = event_type.lower()

    if not RepoManager.get().has_task(task):
        click.echo("no results for the specified task {}, use another task".format(task))
        return

    event_type = EVENT_TYPES.get(event_type, None)
    if event_type is None:
        click.echo("we do not have results for the event type: {}".format(event_type))
        return

    if metric == "avg_loss" or metric == "perplexity":
        result_frame = RepoManager.get().nbest_by_metric(user, metric,
                                                         dataset, task, n, event_type, ascending=True)
    else:
        result_frame = RepoManager.get().nbest_by_metric(user, metric, dataset, task, n, event_type, ascending=False)

    if result_frame is not None:
        click.echo("total {} results found, showing best {} results".format(result_frame.shape[0], n))
        click.echo(result_frame)
    else:
        click.echo("no result found for this query")


# summarize results
@cli.command()
@click.option('--task')
def lbsummary(task):
    """
    Provides a summary of the leaderboard. Options: taskname. If you provide
    a taskname, it will show all users, datasets and metrics for that task.
    This is helpful because we often forget what metrics or datasets were
    used for a task, which are the necessary parameters for the commands
    `results` and `best` and `tasksummary`. Shows
    the summary for all available tasks if no option is specified.
    """
    event_types = [EVENT_TYPES['train'], EVENT_TYPES['valid'], EVENT_TYPES['test']]

    if task is not None:
        if not RepoManager.get().has_task(task):
            click.echo("no results for the specified task {}, use another task".format(task))
        return

    return RepoManager.get().leaderboard_summary(task, event_types, click.echo)


@cli.command()
@click.argument('task')
@click.argument('dataset')
@click.argument('metric')
def tasksummary(task, dataset, metric):
    """
    Provides a natural language summary for a task. This is almost equivalent
    to the `best` command.
    """
    event_type = EVENT_TYPES["test"]
    if not RepoManager.get().has_task(task):
        click.echo("no results for the specified task {}, use another task".format(task))
        return

    task_summary = RepoManager.get().task_summary(task, dataset, metric, event_type)
    if task_summary is None:
        click.echo("can't produce summary for the requested task {}".format(task))
        return

    click.echo(task_summary)


# Edit database
@cli.command()
@click.argument('task')
@click.argument('id')
@click.argument('label')
def updatelabel(id, label, task):
    """Update the _label_ for an experiment (identified by its id) for a task"""
    prev_label, new_label = RepoManager.get().rename_label(id, task, label)
    click.echo("[previous label for the experiment]: {} ".format(prev_label))
    click.echo("[updated label for the experiment]: {} ".format(new_label))


@cli.command()
@click.argument('task')
@click.argument('id')
def delete(id, task):

    prev = RepoManager.get().get_label(id, task)
    if prev is None:
        click.echo("The record {} doesn't exist in the database".format(id))
        return
    click.echo("You are going to delete the record {} " +
        "from {} database. We will also delete the model file if it exists.".format(prev, task))

    if click.confirm('Do you want to continue?'):
        if RepoManager.get().rm(id, task, click.echo) is True:
            click.echo("record {} deleted successfully from database {}".format(id, task))
            return
    click.echo("no record deleted")


# Put results in database
@cli.command()
@click.option("--user", help="username", default=getpass.getuser())
@click.option("--cbase", help="path to the base structure for the model checkpoint files:"
                              "such as ../tagger/tagger-model-tf-11967 or /home/ds/tagger/tagger-model-tf-11967")
@click.option("--cstore", default="/data/model-checkpoints", help="location of the model checkpoint store")
@click.argument('task')
@click.argument('config')
@click.argument('log')
@click.argument('label')
def putresult(user, log, task, config, label, cbase, cstore):

    logf = log.format(task)
    if not os.path.exists(logf):
        click.echo("the log file at {} doesn't exist, provide a valid location".format(logf))
        return
    if not os.path.exists(config):
        click.echo("the config file at {} doesn't exist, provide a valid location".format(config))
        return

    config_file = config
    config_mem = read_config(config_file)
    events_mem = log2json(logf)

    RepoManager.get().put_result(task, config_mem, events_mem,
                                 username=user, label=label, print_fn=click.echo,
                                 checkpoint_base=cbase, checkpoint_store=cstore)


@cli.command()
@click.option("--cstore", default="/data/model-checkpoints", help="location of the model checkpoint store")
@click.argument('task')
@click.argument('id')
@click.argument('cbase')
def putmodel(task, id, cbase, cstore):

    model_loc = RepoManager.get().put_model(id, task, cbase, cstore, click.echo)
    if model_loc is not None:
        click.echo("database updated with {}".format(model_loc))
        return

    click.echo("model could not be stored, see previous errors")


@cli.command()
@click.argument("task")
@click.argument("sha1")
@click.argument("filename")
def config2json(task, sha1, filename):

    if not RepoManager.get().has_task(task):
        click.echo("no results for the specified task {}, use another task".format(task))
        return

    j = RepoManager.get().config2dict(task, sha1)
    if j is None:
        click.echo("can not find config sha1: {}".format(sha1))
        return
    with open(os.path.expanduser(filename), "w") as f:
        json.dump(j, f, indent=True)

if __name__ == "__main__":
    cli()

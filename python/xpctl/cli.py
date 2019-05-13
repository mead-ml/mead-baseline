import sys
import getpass
import json
import os

from click_shell import shell
import click
from xpctl.client import Configuration
from xpctl.client.api import XpctlApi
from xpctl.client import ApiClient
from xpctl.client.rest import ApiException
from xpctl.clihelpers import experiment_to_df, experiment_aggregate_list_to_df, experiment_list_to_df, \
    task_summary_to_df, task_summaries_to_df, read_config_stream
from xpctl.helpers import to_swagger_experiment, store_model
from mead.utils import hash_config
from baseline.utils import read_config_file, write_config_file

EVENT_TYPES = {
    "train": "train_events", "Train": "train_events",
    "test": "test_events", "Test": "test_events",
    "valid": "valid_events", "Valid": "valid_events",
    "dev": "valid_events", "Dev": "valid_events"
}


class ServerManager(object):
    
    api = None
    host = None
    
    @staticmethod
    def get():
        if ServerManager.api is None:
            config = Configuration(host=ServerManager.host)
            api_client = ApiClient(config)
            ServerManager.api = XpctlApi(api_client)
            
        if ServerManager.api is not None:
            click.echo(click.style(
                "connection with xpctl server successful with [host]: {}".format(ServerManager.host), fg='green'
            ))
            return ServerManager.api
        click.echo(click.style("server connection unsuccessful, aborting", fg='red'))
        sys.exit(1)


@shell(prompt="xpctl > ", intro="Starting xpctl...")
@click.option('--host', help="server host", default=None)
@click.option('--config', help="xpctl config", default='~/xpctlcred.json')
def cli(host, config):
    if host is not None:
        ServerManager.host = host
    else:
        ServerManager.host = read_config_file(os.path.expanduser(config))['host']


@cli.command()
@click.argument('task')
@click.argument('eid')
def getmodelloc(task, eid):
    """Get the model location for a particular task and record id"""
    ServerManager.get()
    result = ServerManager.api.get_model_location(task, eid)
    if result.response_type == 'success':
        click.echo(click.style(result.message, fg='green'))
    else:
        click.echo(click.style(result.message, fg='red'))


@cli.command()
@click.option('--event_type', default='test', help="train/ dev/ test")
@click.option('--output', help='output file (csv)', default=None)
@click.option('--metric', multiple=True, help="list of metrics (prec, recall, f1, accuracy),[multiple]: "
                                              "--metric f1 --metric acc", default=None)
@click.option('--sort', help="specify one metric to sort the results", default=None)
@click.option('--output_fields', multiple=True, help="which field(s) you want to see in output",
              default=['username', 'sha1'])
@click.argument('task')
@click.argument('eid')
def experiment(task, eid, event_type, output, metric, sort, output_fields):
    """Get the details for an experiment"""
    event_type = EVENT_TYPES[event_type]
    ServerManager.get()
    try:
        result = ServerManager.api.experiment_details(task, eid, event_type=event_type, metric=metric)
        prop_name_loc = {k: i for i, k in enumerate(output_fields)}
        result_df = experiment_to_df(exp=result, prop_name_loc=prop_name_loc, event_type=event_type, sort=sort)
        if output is None:
            click.echo(result_df)
        else:
            result_df.to_csv(output)
    except ApiException as e:
        click.echo(click.style(json.loads(e.body)['detail'], fg='red'))


@cli.command()
@click.option('--metric', multiple=True, help="list of metrics (prec, recall, f1, accuracy),[multiple]: --metric f1 "
                                              "--metric acc")
@click.option('--sort', help="specify one metric to sort the results")
@click.option('--nconfig', help='number of experiments to aggregate', type=int, default=1)
@click.option('--event_type', default='test', help="train/ dev/ test")
@click.option('--n', help='number of rows to show', type=int, default=-1)
@click.option('--output', help='output file (csv)', default=None)
@click.option('--aggregate_fn', help='aggregate functions', multiple=True,
              type=click.Choice(['min', 'max', 'avg', 'std']), default=['avg', 'std'])
@click.argument('task')
@click.argument('dataset')
def results(task, dataset, metric, sort, nconfig, event_type, n, output, aggregate_fn):
    event_type = EVENT_TYPES[event_type]
    reduction_dim = 'sha1'
    ServerManager.get()
    try:
        result = ServerManager.api.get_results_by_prop(task, prop='dataset', value=dataset, reduction_dim=reduction_dim,
                                                       metric=metric, sort=sort, numexp_reduction_dim=nconfig,
                                                       event_type=event_type)
        result_df = experiment_aggregate_list_to_df(exp_aggs=result, event_type=event_type, aggregate_fns=aggregate_fn)
        if n != -1:
            result_df = result_df.head(n)
        if output is None:
            click.echo(result_df)
        else:
            result_df.to_csv(output)
    except ApiException as e:
        click.echo(click.style(json.loads(e.body)['detail'], fg='red'))


@cli.command()
@click.option('--user', multiple=True, help="list of users (testuser, root), [multiple]: --user a --user b")
@click.option('--metric', multiple=True, help="list of metrics (prec, recall, f1, accuracy),[multiple]: --metric f1 "
                                              "--metric acc")
@click.option('--sort', help="specify one metric to sort the results", default=None)
@click.option('--event_type', default='test', help="specify one metric to sort the results")
@click.option('--n', help='number of experiments', type=int)
@click.option('--output', help='output file (csv)')
@click.option('--output_fields', multiple=True, help="which field(s) you want to see in output",
              default=['username', 'eid'])
@click.argument('task')
@click.argument('sha1')
def details(task, sha1, user, metric, sort, event_type, n, output, output_fields):
    """
    Shows the results for all experiments for a particular config (sha1). Optionally filter out by user(s), metric(s),
    or sort by one metric. Shows the results on the test data by default, provide event_type (train/valid/test)
    to see for other events.
    """
    event_type = EVENT_TYPES[event_type]
    ServerManager.get()
    try:
        result = ServerManager.api.list_experiments_by_prop(task, prop='sha1', value=sha1, user=user, metric=metric,
                                                            sort=sort, event_type=event_type)
        
        prop_name_loc = {k: i for i, k in enumerate(output_fields)}
        result_df = experiment_list_to_df(exps=result, prop_name_loc=prop_name_loc, event_type=event_type)
        if n != -1:
            result_df = result_df.head(n)
        if output is None:
            click.echo(result_df)
        else:
            result_df.to_csv(output)
    except ApiException as e:
        click.echo(click.style(json.loads(e.body)['detail'], fg='red'))


@cli.command()
@click.argument('task')
@click.argument('sha1')
@click.argument('filename')
def config2json(task, sha1, filename):
    """Exports the config file for an experiment as a json file."""
    ServerManager.get()
    try:
        result = ServerManager.api.config2json(task, sha1)
        write_config_file(result, filename)
    except ApiException as e:
        click.echo(click.style(json.loads(e.body)['detail'], fg='red'))


@cli.command()
@click.option('--task')
def lbsummary(task):
    """
    Provides a summary of the leaderboard. Options: taskname. If you provide
    a taskname, it will show all users and datasets for that task.
    This is helpful because we often forget what datasets were
    used for a task, which are the necessary parameters for the commands
    `results` and `best` and `tasksummary`. Shows the summary for all available tasks if no option is specified.
    """
    ServerManager.get()
    try:
        if task is not None:
            result = task_summary_to_df(ServerManager.api.task_summary(task))
        else:
            result = task_summaries_to_df(ServerManager.api.summary())
        click.echo(result)
    except ApiException as e:
        click.echo(click.style(json.loads(e.body)['detail'], fg='red'))


@cli.command()
@click.argument('task')
@click.argument('eid')
@click.argument('label')
def updatelabel(task, label, eid):
    """Update the _label_ for an experiment (identified by its id) for a task"""
    ServerManager.get()
    result = ServerManager.api.update_property(task, eid, prop='label', value=label)
    if result.response_type == 'success':
        click.echo(click.style(result.message, fg='green'))
    else:
        click.echo(click.style(result.message, fg='red'))


@cli.command()
@click.argument('task')
@click.argument('eid')
def delete(task, eid):
    """Deletes a record from the database and the associated model file from model-checkpoints if it exists."""
    ServerManager.get()
    result = ServerManager.api.remove_experiment(task, eid)
    if result.response_type == 'success':
        click.echo(click.style(result.message, fg='green'))
    else:
        click.echo(click.style(result.message, fg='red'))


@cli.command()
@click.option("--user", help="username", default=getpass.getuser())
@click.option("--cbase", help="path to the base structure for the model checkpoint files:"
                              "such as ../tagger/tagger-model-tf-11967 or /home/ds/tagger/tagger-model-tf-11967")
@click.option("--cstore", default="/data/model-checkpoints", help="location of the model checkpoint store")
@click.option("--label", default=None, help="label for the experiment")
@click.argument('task')
@click.argument('config')
@click.argument('log')
def putresult(task, config, log, user, label, cbase, cstore):
    """Puts the results in a database. provide task name, config file, the reporting log file.
    optionally can put the model files in a persistent storage. """
    
    logf = log.format(task)
    if not os.path.exists(logf):
        click.echo(click.style("the log file at {} doesn't exist, provide a valid location".format(logf), fg='red'))
        return
    if not os.path.exists(config):
        click.echo(click.style("the config file at {} doesn't exist, provide a valid location".format(config), fg='red'))
        return
    ServerManager.get()
    result = ServerManager.api.put_result(task, to_swagger_experiment(task, config, log, username=user, label=label))
    if result.response_type == 'success':
        eid = result.message
        click.echo(click.style('results stored with experiment: {}'.format(result.message), fg='green'))
        result = store_model(checkpoint_base=cbase, config_sha1=hash_config(read_config_file(config)),
                             checkpoint_store=cstore, print_fn=click.echo)
        if result is not None:
            click.echo(click.style('model stored at {}'.format(result), fg='green'))
            update_result = ServerManager.api.update_property(task, eid, prop='checkpoint', value=result)
            if update_result.response_type == 'success':
                click.echo(click.style(update_result.message, fg='green'))
            else:
                click.echo(click.style(update_result.message, fg='red'))
        else:
            click.echo(click.style('failed to store model'.format(result), fg='red'))
    else:
        click.echo(click.style(result.message, fg='red'))
        

@cli.command()
@click.option("--cstore", default="/data/model-checkpoints", help="location of the model checkpoint store")
@click.argument('task')
@click.argument('eid')
@click.argument('cbase')
def putmodel(task, eid, cbase, cstore):
    """
    Puts the baseline model files in persistent storage. provide task, id and model base (""tagger-model-tf-").
    optionally provide the storage location (/data/model-checkpoints by default)
    """
    ServerManager.get()
    event_type = EVENT_TYPES['test']
    metric = []
    try:
        result = ServerManager.api.experiment_details(task, eid, event_type=event_type, metric=metric)
        config_obj = read_config_stream(result.config)
        if config_obj is None:
            click.echo('can not process the config for experiment {} in {} database'.format(task, eid))
            sys.exit(1)
        result = store_model(checkpoint_base=cbase, config_sha1=hash_config(config_obj),
                             checkpoint_store=cstore, print_fn=click.echo)
        if result is not None:
            click.echo(click.style('model stored at {}'.format(result), fg='green'))
            update_result = ServerManager.api.update_property(task, eid, prop='checkpoint', value=result)
            if update_result.response_type == 'success':
                click.echo(click.style(update_result.message, fg='green'))
            else:
                click.echo(click.style(update_result.message, fg='red'))
        else:
            click.echo(click.style('failed to store model'.format(result), fg='red'))

    except ApiException as e:
        click.echo(click.style(json.loads(e.body)['detail'], fg='red'))


if __name__ == "__main__":
    cli()

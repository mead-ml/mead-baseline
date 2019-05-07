import sys
import pandas as pd
#import getpass
from click_shell import shell
import click
#from baseline.utils import read_json, read_config_file
from xpctl.swagger_client import Configuration
from xpctl.swagger_client.api import XpctlApi
from xpctl.swagger_client import ApiClient

pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", None)

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
            click.echo("connection with xpctl server successful with [host]: {}".format(ServerManager.host))
            return ServerManager.api
        click.echo("server connection unsuccessful, aborting")
        sys.exit(1)


@shell(prompt="xpctl > ", intro="Starting xpctl...")
@click.option('--host', help="server host", default="localhost:5310/v2")
def cli(host):
    ServerManager.host = host


@cli.command()
@click.argument('task')
@click.argument('id')
def getmodelloc(task, id):
    """Get the model location for a particular task and record id"""
    ServerManager.get()
    result = ServerManager.api.get_model_location(task, id)
    click.echo('[{}] {}'.format(result.response_type, result.message))


#@cli.command()
#@click.option('--metric', multiple=True, help="list of metrics (prec, recall, f1, accuracy),[multiple]: --metric f1 "
#                                               "--metric acc")
# @click.option('--event_type', default='test', help="train/ dev/ test")
# @click.option('--sort', help="specify one metric to sort the results")
# @click.option('--output', help='output file')
# @click.option('--nconfig', help='number of experiments to aggregate', type=int)
# @click.option('--id', help='filter by specific experiment')
# @click.option('--label', help='filter by label')
# @click.option('--n', help='number of rows to show', type=int)
# @click.argument('task')
# @click.argument('dataset')
# def results(metric, event_type, sort, output, nconfig, n, task, dataset, id, label):
#     """
#     Provides a statistical summary of the results for a problem . A problem is defined by a (task, dataset) tuple. For each config used in the task, shows the average, min, max and std dev and number of experiments done using config. Optionally: a. --metrics: choose metric(s) to show. results ll be sorted on the first metric. b. --sort output all metrics but sort on one. c. --n: limit number of experiments. d. event_type: show results for train/dev/valid datasets. e. output: put the results in an output file.
#     """
#     if not ServerManager.get().has_task(task):
#         click.echo("no results for the specified task {}, use another task".format(task))
#         return
#     event_type = EVENT_TYPES[event_type]
#     num_exps = n
#     num_exps_per_config = nconfig
#     results = ServerManager.get().get_results(task, dataset, event_type,
#                                               num_exps, num_exps_per_config,
#                                               metric, sort, id, label)
#     if results is None:
#         click.echo("can't produce summary for the requested task {}".format(task))
#         return
#     click.echo(results)
#     if output is not None:
#         results.to_csv(os.path.expanduser(output), index=True, index_label="config-sha1")
#
#
#
# # summarize results
# @cli.command()
# @click.option('--task')
# def lbsummary(task):
#     """
#     Provides a summary of the leaderboard. Options: taskname. If you provide
#     a taskname, it will show all users and datasets for that task.
#     This is helpful because we often forget what datasets were
#     used for a task, which are the necessary parameters for the commands
#     `results` and `best` and `tasksummary`. Shows the summary for all available tasks if no option is specified.
#     """
#
#     if task is not None:
#         if not ServerManager.get().has_task(task):
#             click.echo("no results for the specified task {}, use another task".format(task))
#             return
#
#     return ServerManager.get().leaderboard_summary(event_type=EVENT_TYPES["test"], task=task, print_fn=click.echo)
#
#
# @cli.command()
# @click.option('--user', multiple=True, help="list of users (testuser, root), [multiple]: --user a --user b")
# @click.option('--metric', multiple=True, help="list of metrics (prec, recall, f1, accuracy),[multiple]: --metric f1 "
#                                               "--metric acc")
# @click.option('--sort', help="specify one metric to sort the results")
# @click.option('--event_type', default='test', help="specify one metric to sort the results")
# @click.option('--n', help='number of experiments', type=int)
# @click.option('--output', help='output file (csv)')
# @click.argument('task')
# @click.argument('sha1')
# def details(user, metric, sort, event_type, task, sha1, n, output):
#     """
#     Shows the results for all experiments for a particular config (sha1). Optionally filter out by user(s), metric(s), or sort by one metric. Shows the results on the test data by default, provide event_type (train/valid/test) to see for other datasets.
#     """
#     if not ServerManager.get().has_task(task):
#         click.echo("no results for the specified task {}, use another task".format(task))
#         return
#
#     event_type = EVENT_TYPES.get(event_type, None)
#     if event_type is None:
#         click.echo("we do not have results for the event type: {}".format(event_type))
#         return
#
#     result_frame = ServerManager.get().experiment_details(user, metric, sort, task, event_type, sha1, n)
#     if result_frame is not None:
#         click.echo(result_frame)
#     else:
#         click.echo("no result found for this query")
#     if output is not None:
#         result_frame.to_csv(os.path.expanduser(output), index=False)
#
#

# # Edit database

@cli.command()
@click.argument('task')
@click.argument('id')
@click.argument('label')
def updatelabel(task, label, id):
    """Update the _label_ for an experiment (identified by its id) for a task"""
    ServerManager.get()
    result = ServerManager.api.update_label(task, id, label)
    click.echo('[{}] {}'.format(result.response_type, result.message))
#
#
# @cli.command()
# @click.argument('task')
# @click.argument('id')
# def delete(id, task):
#     '''
#     Deletes a record from the database and the associated model file from model-checkpoints if it exists.
#     '''
#     prev = ServerManager.get().get_label(id, task)
#     if prev is None:
#         click.echo("The record {} doesn't exist in the database".format(id))
#         return
#     click.echo("You are going to delete the record {} " +
#                "from {} database. We will also delete the model file if it exists.".format(prev, task))
#
#     if click.confirm('Do you want to continue?'):
#         if ServerManager.get().rm(id, task, click.echo) is True:
#             click.echo("record {} deleted successfully from database {}".format(id, task))
#             return
#     click.echo("no record deleted")
#
#
# # Put results in database
# @cli.command()
# @click.option("--user", help="username", default=getpass.getuser())
# @click.option("--cbase", help="path to the base structure for the model checkpoint files:"
#                               "such as ../tagger/tagger-model-tf-11967 or /home/ds/tagger/tagger-model-tf-11967")
# @click.option("--cstore", default="/data/model-checkpoints", help="location of the model checkpoint store")
# @click.option("--label", default=None, help="label for the experiment")
# @click.argument('task')
# @click.argument('config')
# @click.argument('log')
# def putresult(user, cbase, cstore, label, task, config, log):
#     '''
#     Puts the results in a database. provide task name, config file, the reporting log file. optionally can put the model files in a persistent storage.
#     '''
#     logf = log.format(task)
#     if not os.path.exists(logf):
#         click.echo("the log file at {} doesn't exist, provide a valid location".format(logf))
#         return
#     if not os.path.exists(config):
#         click.echo("the config file at {} doesn't exist, provide a valid location".format(config))
#         return
#
#     config_file = config
#     config_mem = read_config_file(config_file)
#     events_mem = log2json(logf)
#
#     ServerManager.get().put_result(task, config_mem, events_mem,
#                                    username=user, label=label, print_fn=click.echo,
#                                    checkpoint_base=cbase, checkpoint_store=cstore)
#
#
# @cli.command()
# @click.option("--cstore", default="/data/model-checkpoints", help="location of the model checkpoint store")
# @click.argument('task')
# @click.argument('id')
# @click.argument('cbase')
# def putmodel(task, id, cbase, cstore):
#     '''
#     Puts the baseline model files in persistent storage. provide task, id and model base (""tagger-model-tf-"). optionally provide the storage location (/data/model-checkpoints by default)
#     '''
#     model_loc = ServerManager.get().put_model(id, task, cbase, cstore, click.echo)
#     if model_loc is not None:
#         click.echo("database updated with {}".format(model_loc))
#         return
#
#     click.echo("model could not be stored, see previous errors")
#
#
# @cli.command()
# @click.argument("task")
# @click.argument("sha1")
# @click.argument("filename")
# def config2json(task, sha1, filename):
#     '''
#     Exports the config file for an experiment as a json file.
#     '''
#     if not ServerManager.get().has_task(task):
#         click.echo("no results for the specified task {}, use another task".format(task))
#         return
#
#     j = ServerManager.get().config2dict(task, sha1)
#     if j is None:
#         click.echo("can not find config sha1: {}".format(sha1))
#         return
#     with open(os.path.expanduser(filename), "w") as f:
#         json.dump(j, f, indent=True)

if __name__ == "__main__":
    cli()
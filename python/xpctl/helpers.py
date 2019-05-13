import os
import shutil
from baseline.utils import unzip_model, read_config_file
import json
from xpctl.client.models import Result, Experiment


def read_logs(file_name):
    logs = []
    with open(file_name) as f:
        for line in f:
            logs.append(json.loads(line))
    return logs


def convert_to_result(event):
    results = []
    non_metrics = ['tick_type', 'tick', 'phase']
    metrics = event.keys() - non_metrics
    for metric in metrics:
        results.append(Result(
            metric=metric,
            value=event[metric],
            tick_type=event['tick_type'],
            tick=event['tick'],
            phase=event['phase']
        )
        )
    return results


def flatten(_list):
    return [item for sublist in _list for item in sublist]


def to_swagger_experiment(task, config, log, **kwargs):
    if type(log) is not str:  # this is a log object and not a file
        events_obj = log
    else:
        events_obj = read_logs(log)
    train_events = flatten(
        [convert_to_result(event) for event in list(filter(lambda x: x['phase'] == 'Train', events_obj))]
    )
    valid_events = flatten(
        [convert_to_result(event) for event in list(filter(lambda x: x['phase'] == 'Valid', events_obj))]
    )
    test_events = flatten(
        [convert_to_result(event) for event in list(filter(lambda x: x['phase'] == 'Test', events_obj))]
    )
    if type(config) is not str:  # this is a config object and not a file
        config = json.dumps(config)
    else:
        config = json.dumps(read_config_file(config))
    d = kwargs
    d.update({'task': task,
              'config': config,
              'train_events': train_events,
              'valid_events': valid_events,
              'test_events': test_events
              })
    
    return Experiment(**d)


def store_model(checkpoint_base, config_sha1, checkpoint_store, print_fn=print):
    checkpoint_base = unzip_model(checkpoint_base)
    mdir, mbase = os.path.split(checkpoint_base)
    mdir = mdir if mdir else "."
    if not os.path.exists(mdir):
        print_fn("no directory found for the model location: [{}], aborting command".format(mdir))
        return None
    
    mfiles = ["{}/{}".format(mdir, x) for x in os.listdir(mdir) if x.startswith(mbase + "-") or
              x.startswith(mbase + ".")]
    if not mfiles:
        print_fn("no model files found with base [{}] at location [{}], aborting command".format(mbase, mdir))
        return None
    model_loc_base = "{}/{}".format(checkpoint_store, config_sha1)
    if not os.path.exists(model_loc_base):
        os.makedirs(model_loc_base)
    dirs = [int(x[:-4]) for x in os.listdir(model_loc_base) if x.endswith(".zip") and x[:-4].isdigit()]
    # we expect dirs in numbers.
    new_dir = "1" if not dirs else str(max(dirs) + 1)
    model_loc = "{}/{}".format(model_loc_base, new_dir)
    os.makedirs(model_loc)
    for mfile in mfiles:
        shutil.copy(mfile, model_loc)
        print_fn("writing model file: [{}] to store: [{}]".format(mfile, model_loc))
    print_fn("zipping model files")
    shutil.make_archive(base_name=model_loc,
                        format='zip',
                        root_dir=model_loc_base,
                        base_dir=new_dir)
    shutil.rmtree(model_loc)
    print_fn("model files zipped and written")
    return model_loc + ".zip"


import json
import os
import shutil
from baseline.utils import export, unzip_model

__all__ = []
exporter = export(__all__)

METRICS_SORT_ASCENDING = ['avg_loss', 'perplexity']


@exporter
def log2json(log_file):
    s = []
    with open(log_file) as f:
        for line in f:
            x = line.replace("'", '"')
            s.append(json.loads(x))
    return s


@exporter
def json2log(events, log_file):
    with open(log_file, 'w') as wf:
        for event in events:
            wf.write(json.dumps(event)+'\n')


@exporter
def get_checkpoint(checkpoint_base, checkpoint_store, config_sha1, hostname):
    if checkpoint_base:
        model_loc = store_model(checkpoint_base, config_sha1, checkpoint_store)
        if model_loc is not None:
            return "{}:{}".format(hostname, os.path.abspath(model_loc))
        else:
            raise RuntimeError("model could not be stored, see previous errors")


@exporter
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


@exporter
def get_experiment_label(config_obj, task, **kwargs):
    if kwargs.get('label', None) is not None:
        return kwargs['label']
    if 'description' in config_obj:
        return config_obj['description']
    else:
        model_type = config_obj.get('model_type', 'default')
        backend = config_obj.get('backend', 'tensorflow')
        return "{}-{}-{}".format(task, backend, model_type)

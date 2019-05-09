import json
import os
from baseline.utils import export

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
def get_experiment_label(config_obj, task, **kwargs):
    if kwargs.get('label', None) is not None:
        return kwargs['label']
    if 'description' in config_obj:
        return config_obj['description']
    else:
        model_type = config_obj.get('model_type', 'default')
        backend = config_obj.get('backend', 'tensorflow')
        return "{}-{}-{}".format(task, backend, model_type)

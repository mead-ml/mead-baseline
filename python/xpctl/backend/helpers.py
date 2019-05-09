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

# def sort_ascending(metric):
#     return metric == "avg_loss" or metric == "perplexity"
#
#
# def df_summary_exp(df):
#     return df.groupby("sha1").agg([len, np.mean, np.std, np.min, np.max]) \
#         .rename(columns={'len': 'num_exps', 'amean': 'mean', 'amin': 'min', 'amax': 'max'})
#
#
# def df_get_results(result_frame, dataset, num_exps, num_exps_per_config, metric, sort):
#     datasets = result_frame.dataset.unique()
#     if dataset not in datasets:
#         return None
#     dsr = result_frame[result_frame.dataset == dataset]
#     if dsr.empty:
#         return None
#     df = pd.DataFrame()
#     if num_exps_per_config is not None:
#         for gname, rframe in result_frame.groupby("sha1"):
#             rframe = rframe.copy()
#             rframe['date'] =pd.to_datetime(rframe.date)
#             rframe = rframe.sort_values(by='date', ascending=False).head(int(num_exps_per_config))
#             df = df.append(rframe)
#         result_frame = df
#
#     result_frame = result_frame.drop(["id"], axis=1)
#     result_frame = result_frame.groupby("sha1").agg([len, np.mean, np.std, np.min, np.max])\
#         .rename(columns={'len': 'num_exps', 'amean': 'mean', 'amin': 'min', 'amax': 'max'})
#     metrics = listify(metric)
#     if len(metrics) == 1:
#         result_frame = result_frame.sort_values([(metrics[0], 'mean')], ascending=sort_ascending(metric))
#     if sort:
#         result_frame = result_frame.sort_values([(sort, 'mean')], ascending=sort_ascending(metric))
#     if result_frame.empty:
#         return None
#     if num_exps is not None:
#         result_frame = result_frame.head(num_exps)
#     return result_frame
#
#
# def df_experimental_details(result_frame, sha1, users, sort, metric, num_exps):
#     result_frame = result_frame[result_frame.sha1 == sha1]
#     if result_frame.empty:
#         return None
#     if users is not None:
#         df = pd.DataFrame()
#         for user in users:
#             df = df.append(result_frame[result_frame.username == user])
#         result_frame = result_frame
#     metrics = list(metric)
#     if len(metrics) == 1:
#         result_frame = result_frame.sort_values([metrics[0]], ascending=sort_ascending(metric))
#     if sort:
#         result_frame = result_frame.sort_values([sort], ascending=sort_ascending(metric))
#     if result_frame.empty:
#         return None
#     if num_exps is not None:
#         result_frame = result_frame.head(num_exps)
#     return result_frame
#
#
#
#
# def aggregate_results(resultset, groupby_key, num_exps_per_reduction, num_exps):
#     grouped_result = resultset.groupby(groupby_key)
#
#     aggregate_fns = {'min': np.min, 'max': np.max, 'avg': np.mean, 'std': np.std}
#
#     return grouped_result.reduce(aggregate_fns=aggregate_fns)
#
#

import argparse
import mead
from mead.utils import convert_path
from baseline.utils import unzip_model
from baseline.utils import read_config_file
from mead.exporters import create_exporter

def main():
    parser = argparse.ArgumentParser(description='Train a text classifier')
    parser.add_argument('--config', help='JSON Configuration for an experiment', required=True, type=convert_path)
    parser.add_argument('--settings', help='JSON Configuration for mead', required=False, default='config/mead-settings.json', type=convert_path)
    parser.add_argument('--datasets', help='json library of dataset labels', default='config/datasets.json', type=convert_path)
    parser.add_argument('--embeddings', help='json library of embeddings', default='config/embeddings.json', type=convert_path)
    parser.add_argument('--logging', help='json file for logging', default='config/logging.json', type=convert_path)
    parser.add_argument('--task', help='task to run', choices=['classify', 'tagger', 'seq2seq', 'lm'])
    parser.add_argument('--exporter_type', help='exporter type', default='default')
    parser.add_argument('--model', help='model name', required=True, type=unzip_model)
    parser.add_argument('--model_version', help='model_version', default=1)
    parser.add_argument('--output_dir', help='output dir', default='./models')
    parser.add_argument('--no_preproc', help="skip preprocessing", dest='use_preproc', action='store_false')
    args = parser.parse_args()

    config_params = read_config_file(args.config)
    task_name = config_params.get('task', 'classify') if args.task is None else args.task
    task = mead.Task.get_task_specific(task_name, args.logging, args.settings)
    task.read_config(config_params, args.datasets)
    exporter = create_exporter(task, args.exporter_type)

    embeddings_set = mead.utils.read_config_file(args.embeddings)
    embeddings_set = mead.utils.index_by_label(embeddings_set)

    features = list_model_features(args.model + ".state")
    feature_descs = {}
    for feat, ty in features.items():
        f = get_feature_md_object(args.model, feat)
        f.update({
            'type': ty
        })
        feature_descs[feat] = f

    exporter.run(args.model, embeddings_set, feature_descs, args.output_dir, args.model_version, args.use_preproc)

def list_model_features(model_state_file):
    """
    read what features are expected for this model from the state file.

    typically this means `word` and `char`.
    """
    state = mead.utils.read_config_file(model_state_file)
    if 'embeddings' not in state:
        return []

    return state['embeddings']

def get_feature_md_object(model_name, feature):
    """
    return the feature object from the md json file that was saved during training.
    """
    feat_file = '-'.join([model_name, feature, 'md']) + '.json'
    return mead.utils.read_config_file(feat_file)




if __name__ == "__main__":
    main()

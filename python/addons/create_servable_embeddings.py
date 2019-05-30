import os
import logging
import argparse
from copy import deepcopy
from itertools import chain
from collections import defaultdict
import tensorflow as tf
from baseline.version import __version__
from baseline.model import create_model_for, load_model_for, register_model
from baseline.utils import (
    str2bool,
    zip_files,
    read_json,
    write_json,
    MAGIC_VARS,
    save_vocabs,
    get_model_file,
    save_vectorizers,
    normalize_backend,
    read_config_stream,
)
from baseline.tf.embeddings import LargeLookupTableEmbeddings
from baseline.tf.tfy import (
    tf_device_wrapper,
    reload_embeddings,
    new_placeholder_dict,
    create_session
)
import mead
from mead.tasks import Task, register_task, Backend
from mead.utils import (
    convert_path,
    index_by_label,
    get_export_params,
    read_config_file_or_json,
    create_feature_exporter_field_map,
)
from mead.exporters import create_exporter, register_exporter
from mead.tf.preproc_exporters import PreProcessorController, peek_lengths_key
from mead.tf.exporters import SignatureOutput, create_assets, TensorFlowExporter


logger = logging.getLogger('mead')
bl_logger = logging.getLogger('baseline')


@register_task
class ServableEmbeddingsTask(Task):
    def __init__(self, mead_settings_config, **kwargs):
        super(ServableEmbeddingsTask, self).__init__(mead_settings_config, **kwargs)

    @classmethod
    def task_name(cls):
        return 'servable_embeddings'

    def read_config(self, config_params, datasets_index, **kwargs):
        self.config_params = config_params
        basedir = self.get_basedir()
        if basedir is not None and not os.path.exists(basedir):
            logger.info('Creating: %s', basedir)
            os.makedirs(basedir)
        self._setup_task(**kwargs)
        self._load_user_modules()
        self._create_vectorizers()

    def _setup_task(self, **kwargs):
        self.backend = Backend(self.config_params.get('backend', 'tf'))
        self.backend.load()

    def initialize(self, embeddings):
        embeddings = read_config_file_or_json(embeddings, 'embeddings')
        embeddings_set = index_by_label(embeddings)
        self.config_params['keep_unused'] = True
        features = self.config_params['features']
        self.embeddings, self.feat2index = self._create_embeddings(
            embeddings_set, defaultdict(dict), self.config_params['features']
        )
        save_vocabs(self.get_basedir(), self.feat2index)

    def train(self, checkpoint=None, zip_model=True):
        save_vectorizers(self.get_basedir(), self.vectorizers)
        basename = get_model_file(self.task_name(), self.backend.name, self.get_basedir())
        model = create_servable_embeddings_model(self.embeddings, **self.config_params.get('model', {}))
        model.save(basename)
        if zip_model:
           zip_files(self.get_basedir())
        return None, None


class ServableEmbeddingsModel(object):
    task_name = 'servable_embeddings'

    def __init__(self):
        super(ServableEmbeddingsModel, self).__init__()

    def save(self, basename):
        pass

    @classmethod
    def load(cls, basename, **kwargs):
        pass

    def predict(self, batch_dict):
        pass

def create_servable_embeddings_model(embeddings, **kwargs):
    return create_model_for('servable_embeddings', embeddings, None, **kwargs)


def load_servable_embeddings_model(filename, **kwargs):
    return load_model_for('servable_embeddings', filename, **kwargs)


@register_model(task='servable_embeddings', name='default')
class ServableTensorFlowEmbeddingsModel(ServableEmbeddingsModel):

    def __init__(self):
        super(ServableTensorFlowEmbeddingsModel, self).__init__()
        self._unserializable = []

    @classmethod
    def create(cls, embeddings, **kwargs):
        sess = kwargs.get('sess', create_session())
        model = cls()
        model.embeddings = embeddings
        model._record_state(**kwargs)
        model.embedded = model.embed(**kwargs)
        model.sess = sess
        model.saver = kwargs.get('saver', tf.train.Saver())
        feed_dict = {k: v for e in embeddings.values() for k, v in e.get_feed_dict().items()}
        if kwargs.get('init', True):
            # If we have any luts that are large be sure to fill the embeddings
            # With the weight values on initialization.
            model.sess.run(tf.global_variables_initializer(), feed_dict)
        return model

    def _record_state(self, **kwargs):
        embeddings_info = {}
        for k, v in self.embeddings.items():
            embeddings_info[k] = v.__class__.__name__

        blacklist = set(chain(self._unserializable, MAGIC_VARS, self.embeddings.keys()))
        self._state = {k: v for k, v in kwargs.items() if k not in blacklist}
        self._state.update({
            'version': __version__,
            'module': self.__class__.__module__,
            'class': self.__class__.__name__,
            'embeddings': embeddings_info,
        })

    @classmethod
    @tf_device_wrapper
    def load(cls, basename, **kwargs):
        _state = read_json("{}.state".format(basename))
        if __version__ != _state['version']:
            bl_logger.warning("Loaded model is from baseline version %s, running version is %s", _state['version'], __version__)
        _state['sess'] = kwargs.pop('sess', create_session())
        with _state['sess'].graph.as_default():
            embeddings_info = _state.pop('embeddings')
            embeddings = reload_embeddings(embeddings_info, basename)
            for k in embeddings_info:
                if k in kwargs:
                    _state[k] = kwargs[k]
            model = cls.create(embeddings, init=kwargs.get('init', True), **_state)
            model._state = _state
            model.saver = tf.train.Saver()
            model.saver.restore(model.sess, basename)
        return model

    def embed(self, **kwargs):
        out = []
        for k, embed in self.embeddings.items():
            x = kwargs.get(k, None)
            out.append(embed.encode(x))
        return tf.concat(values=out, axis=-1)

    def make_input(self, batch_dict):
        feed_dict = new_placeholder_dict(False)
        for k in self.embeddings.keys():
            feed_dict["{}:0".format(k)] = batch_dict[k]
        return feed_dict

    def predict(self, batch_dict):
        feed_dict = self.make_input(batch_dict)
        embedded = self.sess.run(self.embedded, feed_dict=feed_dict)
        return embedded

    def save(self, basename, **kwargs):
        self.save_md(basename)
        self.save_values(basename)

    def save_md(self, basename):
        write_json(self._state, '{}.state'.format(basename))
        for k, e in self.embeddings.items():
            e.save_md('{}-{}-md.json'.format(basename, k))

    def save_values(self, basename):
        self.saver.save(self.sess, basename)


@register_exporter(task='servable_embeddings', name='default')
class ServableEmbeddingsTensorFlowExporter(TensorFlowExporter):
    def __init__(self, task, **kwargs):
        super(ServableEmbeddingsTensorFlowExporter, self).__init__(task, **kwargs)

    def _create_model(self, sess, basename, **kwargs):
        model = load_servable_embeddings_model(basename, sess=sess, **kwargs)
        return model, model.embedded

    def _create_rpc_call(self, sess, basename, **kwargs):
        model, embedded = self._create_model(sess, basename)

        predict_tensors = {}
        for k, v in model.embeddings.items():
            try:
                predict_tensors[k] = tf.saved_model.utils.build_tensor_info(v.x)
            except:
                raise Exception('Unknown attribute in signature: {}'.format(v))

        sig_input = predict_tensors
        classes = tf.ones([1, 1], dtype=tf.uint8)
        sig_output = SignatureOutput(classes, embedded)
        sig_name = 'embed_text'

        assets = create_assets(basename, sig_input, sig_output, sig_name, None, return_labels=False)
        return sig_input, sig_output, sig_name, assets


@register_exporter(task='servable_embeddings', name='preproc')
class ServableEmbeddingsTensorFlowPreProcExporter(ServableEmbeddingsTensorFlowExporter):

    def __init__(self, task, **kwargs):
        super(ServableEmbeddingsTensorFlowPreProcExporter, self).__init__(task, **kwargs)
        self.feature_exporter_field_map = kwargs.get('feature_exporter_field_map', {'tokens': 'text'})

    def _create_rpc_call(self, sess, model_file, **kwargs):
        model_base_dir = os.path.split(model_file)[0]
        pid = model_file.split("-")[-1]
        lengths = peek_lengths_key(model_file, self.feature_exporter_field_map)
        pc = PreProcessorController(model_base_dir, pid, self.task.config_params['features'],
                                    self.feature_exporter_field_map, lengths)
        tf_example, preprocessed = pc.run()
        # Create a dict of embedding names to sub-graph outputs to wire in as embedding inputs
        embedding_inputs = {}
        for feature in preprocessed:
            embedding_inputs[feature] = preprocessed[feature]

        model, embedded = self._create_model(sess, model_file, **embedding_inputs)

        sig_input = {x: tf.saved_model.utils.build_tensor_info(tf_example[x]) for x in pc.FIELD_NAMES}
        classes = tf.ones([1, 1], dtype=tf.uint8)
        sig_output = SignatureOutput(classes, embedded)
        sig_name = 'embed_text'
        assets = create_assets(model_file, sig_input, sig_output, sig_name, None,
                               return_labels=False)
        return sig_input, sig_output, sig_name, assets



def main():
    parser = argparse.ArgumentParser(description='Create an Embeddings Service')
    parser.add_argument('--config', help='JSON Configuration for an experiment', type=convert_path, default="$MEAD_CONFIG")
    parser.add_argument('--settings', help='JSON Configuration for mead', default='config/mead-settings.json', type=convert_path)
    parser.add_argument('--datasets', help='json library of dataset labels', default='config/datasets.json', type=convert_path)
    parser.add_argument('--embeddings', help='json library of embeddings', default='config/embeddings.json', type=convert_path)
    parser.add_argument('--backend', help='The deep learning backend to use')
    parser.add_argument('--export', help='Should this create a export bundle?', default=True, type=str2bool)
    parser.add_argument('--exporter_type', help="exporter type (default 'default')", default=None)
    parser.add_argument('--model_version', help='model_version', default=None)
    parser.add_argument('--output_dir', help="output dir (default './models')", default=None)
    parser.add_argument('--project', help='Name of project, used in path first', default=None)
    parser.add_argument('--name', help='Name of the model, used second in the path', default=None)
    parser.add_argument('--is_remote', help='if True, separate items for remote server and client. If False bundle everything together (default True)', default=None)
    args, reporting_args = parser.parse_known_args()

    config_params = read_config_stream(args.config)
    try:
        args.settings = read_config_stream(args.settings)
    except:
        logger.warning('Warning: no mead-settings file was found at [{}]'.format(args.settings))
        args.settings = {}
    args.datasets = read_config_stream(args.datasets)
    args.embeddings = read_config_stream(args.embeddings)

    if args.backend is not None:
        config_params['backend'] = normalize_backend(args.backend)

    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    os.environ['NV_GPU'] = ""
    if 'gpus' in config_params.get('train', {}):
        del config_params['train']['gpus']

    config_params['task'] = 'servable_embeddings'
    task = mead.Task.get_task_specific(config_params['task'], args.settings)
    task.read_config(config_params, args.datasets, reporting_args=[], config_file=deepcopy(config_params))
    task.initialize(args.embeddings)

    to_zip = False if args.export else True
    task.train(None, zip_model=to_zip)

    if args.export:
        model = os.path.abspath(task.get_basedir())
        output_dir, project, name, model_version, exporter_type, return_labels, is_remote = get_export_params(
            config_params.get('export', {}),
            args.output_dir,
            args.project, args.name,
            args.model_version,
            args.exporter_type,
            False,
            args.is_remote,
        )
        feature_exporter_field_map = create_feature_exporter_field_map(config_params['features'])
        exporter = create_exporter(task, exporter_type, return_labels=return_labels,
                                   feature_exporter_field_map=feature_exporter_field_map)
        exporter.run(model, output_dir, project, name, model_version, remote=is_remote)


if __name__ == "__main__":
    main()

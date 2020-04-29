import os
import json
import baseline
from mead.exporters import register_exporter
from mead.preprocessors import create_preprocessors
from baseline.tf.embeddings import *
from baseline.utils import exporter, read_json
from collections import namedtuple
from mead.tf.exporters import ClassifyTensorFlowExporter, TaggerTensorFlowExporter, create_assets

__all__ = []
export = exporter(__all__)

SignatureOutput = namedtuple("SignatureOutput", ("classes", "scores"))


def get_string_size(tensor):
    """Get the number of tokens in a string sperated by space."""
    split = tf.string_split(tf.reshape(tensor, [-1]))
    return tf.shape(split)[1]


def get_mxlen(tensors):
    """Find the longest string in a batch."""
    sizes = tf.map_fn(
        get_string_size, tensors,
        dtype=tf.int32,
        back_prop=False
    )
    return tf.reduce_max(sizes)

def get_lengths(tensors):
    return tf.map_fn(
        get_string_size, tensors, dtype=tf.int32, back_prop=False
    )


def get_lengths(tensors):
    return tf.map_fn(
        get_string_size, tensors, dtype=tf.int32, back_prop=False
    )

def peek_lengths_key(model_file, field_map):
    """Check if there is lengths key to use when finding the length. (defaults to tokens)"""
    peek_state = read_json(model_file + ".state")
    lengths = peek_state.get('lengths_key', 'tokens')
    lengths = lengths.replace('_lengths', '')
    return field_map.get(lengths, lengths)


class PreProcessorController(object):
    def __init__(self, model_base_dir, pid, features, feature_exporter_field_map, length_field='tokens'):
        self.length_field = length_field
        saved_vectorizers = self.get_vectorizers(model_base_dir, pid)
        feature_names = [feature['name'] for feature in features]
        feature_vectorizer_mapping = {feature['name']: feature['vectorizer']['type'] for feature in features}
        self.preprocessors = dict()
        indices, vocabs = self.create_vocabs(model_base_dir, pid, feature_names)
        self.feature_exporter_field_map = feature_exporter_field_map
        self.FIELD_NAMES = list(set(self.feature_exporter_field_map.values()))
        for feature in feature_vectorizer_mapping:
            self.preprocessors[feature] = create_preprocessors(preprocessor_type=feature_vectorizer_mapping[feature],
                                                               feature=feature,
                                                               vectorizer=saved_vectorizers[feature],
                                                               index=indices[feature],
                                                               vocab=vocabs[feature])

    @staticmethod
    def get_vectorizers(model_base_dir, pid):
        """
        :model_file the path-like object to the model and model name.
        :vocab_suffixes the list of vocab types. e.g. 'word', 'char', 'ner'.
        """
        import pickle
        return pickle.load(open(os.path.join(model_base_dir, "vectorizers-{}.pkl".format(pid)), "rb"))

    def _create_example(self):
        serialized_tf_example = tf.compat.v1.placeholder(tf.string, name='tf_example')
        feature_configs = {f: tf.FixedLenFeature(shape=[], dtype=tf.string) for f in self.FIELD_NAMES}
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        return tf_example

    def preproc(self, tf_example):
        preprocessed_inputs = {}
        for feature in self.preprocessors:
            preprocessed_inputs[feature] = self.preprocessors[feature].preproc(tf_example, self.mxlen)
        return preprocessed_inputs

    def create_preprocessed_input(self, tf_example):
        """
        Create a preprocessor chain inside of the tensorflow graph.
        """
        self.mxlen = get_mxlen(tf_example[self.length_field])

        types = {f: tf.int64 for f in self.preprocessors}
        return tf.map_fn(
            self.preproc, tf_example,
            dtype=types, back_prop=False
        )

    def create_vocabs(self, model_base_dir, pid, features):
        """
        :model_file the path-like object to the model and model name.
        :vocab_suffixes the list of vocab types. e.g. 'word', 'char', 'ner'.
        """
        indices = {}
        vocabs = {}
        for feature in features:
            feature_vocab_file = os.path.join(model_base_dir, "vocabs-{}-{}.json".format(feature, pid))
            if os.path.exists(feature_vocab_file):
                indices[feature], vocabs[feature] = self._read_vocab(feature_vocab_file, feature)
        return indices, vocabs

    @staticmethod
    def _read_vocab(vocab_file, feature_name):
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)

        # Make a vocab list
        vocab_list = [''] * (len(vocab) + 1)

        for v, i in vocab.items():
            vocab_list[i] = v

        tok2index = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(vocab_list),
            default_value=Offsets.UNK,
            dtype=tf.string,
            name='%s2index' % feature_name
        )
        return tok2index, vocab

    def run(self):
        tf_example = self._create_example()
        preprocessed = self.create_preprocessed_input(tf_example)
        lengths = get_lengths(tf_example[self.length_field])
        return tf_example, preprocessed, lengths


@export
@register_exporter(task='classify', name='preproc')
class ClassifyTensorFlowPreProcExporter(ClassifyTensorFlowExporter):

    @classmethod
    def preproc_type(cls):
        return 'server'

    def __init__(self, task, **kwargs):
        super(ClassifyTensorFlowPreProcExporter, self).__init__(task, **kwargs)
        self.feature_exporter_field_map = kwargs.get('feature_exporter_field_map', {'tokens': 'text'})
        self.return_labels = kwargs.get('return_labels', False)

    def _create_rpc_call(self, sess, model_file, **kwargs):
        model_base_dir = os.path.split(model_file)[0]
        pid = model_file.split("-")[-1]

        lengths = peek_lengths_key(model_file, self.feature_exporter_field_map)

        pc = PreProcessorController(model_base_dir, pid, self.task.config_params['features'],
                                    self.feature_exporter_field_map, lengths)
        tf_example, preprocessed, lengths = pc.run()
        # Create a dict of embedding names to sub-graph outputs to wire in as embedding inputs
        embedding_inputs = {}
        for feature in preprocessed:
            embedding_inputs[feature] = preprocessed[feature]
        model, classes, values = self._create_model(sess, model_file, lengths=lengths, **embedding_inputs)
        sig_input = {x: tf.saved_model.utils.build_tensor_info(tf_example[x]) for x in pc.FIELD_NAMES}
        if model.lengths is not None:
            sig_input.update({model.lengths_key: tf.saved_model.utils.build_tensor_info(model.lengths)})
        sig_output = SignatureOutput(classes, values)
        sig_name = 'predict_text'
        assets = create_assets(
            model_file,
            sig_input, sig_output, sig_name,
            model.lengths_key,
            return_labels=self.return_labels,
            preproc=self.preproc_type(),
        )
        return sig_input, sig_output, sig_name, assets


@export
@register_exporter(task='tagger', name='preproc')
class TaggerTensorFlowPreProcExporter(TaggerTensorFlowExporter):

    @classmethod
    def preproc_type(cls):
        return 'server'

    def __init__(self, task, **kwargs):
        super(TaggerTensorFlowPreProcExporter, self).__init__(task, **kwargs)
        self.feature_exporter_field_map = kwargs.get('feature_exporter_field_map', {'tokens': 'text'})
        self.return_labels = kwargs.get('return_labels', False)

    def _create_rpc_call(self, sess, model_file, **kwargs):
        model_base_dir = os.path.split(model_file)[0]
        pid = model_file.split("-")[-1]
        lengths = peek_lengths_key(model_file, self.feature_exporter_field_map)
        pc = PreProcessorController(model_base_dir, pid, self.task.config_params['features'],
                                    self.feature_exporter_field_map, lengths)
        tf_example, preprocessed, lengths = pc.run()
        # Create a dict of embedding names to sub-graph outputs to wire in as embedding inputs
        embedding_inputs = {}
        for feature in preprocessed:
            embedding_inputs[feature] = preprocessed[feature]
        model, classes, values = self._create_model(sess, model_file, lengths=lengths, **embedding_inputs)
        sig_input = {x: tf.saved_model.utils.build_tensor_info(tf_example[x]) for x in pc.FIELD_NAMES}
        sig_input.update({model.lengths_key: tf.saved_model.utils.build_tensor_info(model.lengths)})
        sig_output = SignatureOutput(classes, values)
        sig_name = 'tag_text'
        assets = create_assets(
            model_file,
            sig_input, sig_output, sig_name,
            model.lengths_key,
            return_labels=self.return_labels,
            preproc=self.preproc_type(),
        )
        return sig_input, sig_output, sig_name, assets

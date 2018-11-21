import numpy as np
import tensorflow as tf
import json
import baseline
import os
from mead.exporters import register_exporter
import mead.utils
import mead.exporters
from mead.tf.exporters import TensorFlowExporter, create_bundle, create_assets, create_metadata
from mead.tf.preprocessor import PreprocessorCreator
from baseline.utils import export
from baseline.tf.tfy import get_vocab_file_suffixes
from collections import namedtuple
from baseline.utils import ls_props, read_json


FIELD_NAME = 'text/tokens'

__all__ = []
exporter = export(__all__)
SignatureOutput = namedtuple("SignatureOutput", ("classes", "scores"))


@exporter
class TensorFlowPreProcExporter(TensorFlowExporter):
    DEFAULT_VOCABS = {"word", "char"}

    def __init__(self, task):
        super(TensorFlowPreProcExporter, self).__init__(task)

    def get_raw_post(self, tf_example):
        return tf_example[FIELD_NAME]

    def run(self, basename, output_dir, model_version, **kwargs):
        embeddings = kwargs.get('embeddings')
        embeddings_set = mead.utils.read_config_file(embeddings)
        embeddings_set = mead.utils.index_by_label(embeddings_set)
        with tf.Graph().as_default():
            config_proto = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config_proto) as sess:
                sig_input, sig_output, sig_name, assets = self._create_rpc_call(sess, basename, embeddings_set=embeddings_set)

                output_path = os.path.join(output_dir, str(model_version))
                print('Exporting trained model to %s' % output_path)

                try:
                    builder = self._create_saved_model_builder(sess, output_path, sig_input, sig_output, sig_name)
                    create_bundle(builder, output_path, basename, assets)
                    print('Successfully exported model to %s' % output_dir)
                except AssertionError as e:
                    # model already exists
                    raise e
                except Exception as e:
                    # export process broke.
                    # TODO(MB): we should remove the directory, if one has been saved already.
                    raise e

    @staticmethod
    def read_vocab(basename, ty):
        if ty is None:
            vocab_file = basename
            ty = 'word'
        else:
            vocab_file = "{}-{}.vocab".format(basename, ty)
        print('Reading {}'.format(vocab_file))
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)

        # Make a vocab list
        vocab_list = [''] * (len(vocab) + 1)

        for v, i in vocab.items():
            vocab_list[i] = v

        tok2index = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(vocab_list),
            default_value=0,
            dtype=tf.string,
            name='%s2index' % ty
        )
        return tok2index, vocab

    def load_labels(self, basename):

        label_file = '%s.labels' % basename
        with open(label_file, 'r') as f:
            labels = json.load(f)
        return labels

    def _create_example(self):
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        tf_example = tf.parse_example(serialized_tf_example)
        return serialized_tf_example, tf_example

    def _create_vocabs(self, model_file):
        """
        :model_file the path-like object to the model and model name.
        :vocab_suffixes the list of vocab types. e.g. 'word', 'char', 'ner'.
        """
        vocabs = {}
        indices = {}
        if os.path.exists(model_file + '.vocab'):
            indices['word'], vocabs['word'] = TensorFlowExporter.read_vocab(model_file + '.vocab', ty=None)
        else:
            vocab_suffixes = get_vocab_file_suffixes(model_file)
            for suffix in vocab_suffixes:
                indices[suffix], vocabs[suffix] = TensorFlowExporter.read_vocab(model_file, suffix)

        return indices, vocabs

    def assign_char_lookup(self):
        upchars = tf.constant([chr(i) for i in range(65, 91)])
        self.lchars = tf.constant([chr(i) for i in range(97, 123)])
        self.upchars_lut = tf.contrib.lookup.index_table_from_tensor(mapping=upchars, num_oov_buckets=1, default_value=-1)

    def _initialize_embeddings_map(self, vocabs, embeddings_set):
        """
        generate a mapping of vocab_typ (word, char) to the embedding object.
        """
        embeddings = {}

        for vocab_type in vocabs.keys():
            dimension_size = self._get_embedding_dsz(embeddings_set, vocab_type)
            embeddings[vocab_type] = self._initialize_embedding(dimension_size, vocabs[vocab_type])

        return embeddings

    def _get_embedding_dsz(self, embeddings_set, embed_type):
        if embed_type == 'word':
            word_embeddings = self.task.config_params["word_embeddings"]
            return embeddings_set[word_embeddings["label"]]["dsz"]
        elif embed_type == 'char':
            return self.task.config_params["charsz"]

    def _initialize_embedding(self, dimensions_size, vocab):
        return baseline.RandomInitVecModel(dimensions_size, vocab, False)

    def _run_preproc(self, model_params, vocabs, model_file, indices):
        serialized_tf_example, tf_example = self._create_example()
        raw_posts = tf_example[FIELD_NAME]

        preprocessed, lengths = self._create_preprocessed_input(tf_example,
                                                                model_file,
                                                                indices)

        model_params["x"] = preprocessed['word']
        if 'char' in vocabs:
            model_params['xch'] = preprocessed['char']

        return serialized_tf_example, tf_example, raw_posts, lengths

    def _create_preprocessed_input(self, tf_example,
                                   model_file,
                                   indices):
        """
        Create a preprocessor chain inside of the tensorflow graph.
        """
        mxlen, mxwlen = self._get_max_lens(model_file)

        preprocessor = PreprocessorCreator(
            indices, self.lchars, self.upchars_lut, self.task, FIELD_NAME, mxlen, mxwlen)
        types = {k: tf.int64 for k in indices}
        preprocessed, lengths = tf.map_fn(
            preprocessor.preproc_post, tf_example,
            dtype=(types, tf.int32), back_prop=False
        )

        return preprocessed, lengths

    def _get_max_lens(self, base_name):
        mxlen = self.task.config_params['preproc']['mxlen']
        mxwlen = self.task.config_params['preproc'].get('mxwlen')
        state = baseline.utils.read_json("{}.state".format(base_name))
        if 'mxlen' in state:
            mxlen = state['mxlen']
        # What should be called mxwlen is called maxw in the state object of this is for backwards compatibility.
        if 'maxw' in state:
            mxwlen = state['maxw']
        if 'mxwlen' in state:
            mxwlen = state['mxwlen']
        return mxlen, mxwlen


@exporter
@register_exporter(task='classify-preproc', name='default')
class ClassifyTensorFlowPreProcExporter(TensorFlowPreProcExporter):

    def __init__(self, task):
        super(ClassifyTensorFlowPreProcExporter, self).__init__(task)

    def _create_model(self, sess, basename, **kwargs):
        # get embeddings
        indices, vocabs = self._create_vocabs(basename)
        embeddings_set = kwargs.get('embeddings_set')
        labels = self.load_labels(basename)
        embeddings = self._initialize_embeddings_map(vocabs, embeddings_set)

        # Read the state file
        state = read_json(basename + '.state')

        model_params = self.task.config_params["model"]

        model_params["sess"] = sess
        print(model_params)

        # Instantiate a graph
        model = baseline.model.create_model_for(self.task.task_name(), embeddings, labels, **model_params)

        # Set the properties of the model from the state file
        for prop in ls_props(model):
            if prop in state:
                setattr(model, prop, state[prop])

        # Append to the graph for class output
        values, indices = tf.nn.top_k(model.probs, len(labels))
        class_tensor = tf.constant(model.labels)
        table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
        classes = table.lookup(tf.to_int64(indices))
        # Restore the checkpoint
        self._restore_checkpoint(sess, basename)
        return model, classes, values

    def _create_rpc_call(self, sess, basename, **kwargs):
        model, classes, values = self._create_model(sess, basename)
        indices, vocabs = self._create_vocabs(basename)
        self.assign_char_lookup()
        model_params = self.task.config_params["model"]

        serialized_tf_example, tf_example, raw_posts, lengths = self._run_preproc(model_params,
                                                                                  vocabs,
                                                                                  basename,
                                                                                  indices)
        sig_input = {} #SignatureInput(serialized_tf_example, tf_example)

        sig_output = SignatureOutput(classes, values)
        sig_name = 'predict_text'
        assets = create_assets(basename, sig_input, sig_output, sig_name, model.lengths_key)
        return sig_input, sig_output, sig_name, assets



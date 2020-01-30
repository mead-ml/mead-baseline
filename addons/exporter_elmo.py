from mead.tf.exporters import TensorFlowExporter, PreprocessorCreator
from mead.tf.signatures import SignatureInput, SignatureOutput
import tensorflow as tf
import baseline
import numpy as np
import os
FIELD_NAME = 'text/tokens'

class ElmoPreprocessorCreator(PreprocessorCreator):
    def __init__(self, indices, lchars, upchars_lut, task, token_key, extra_feats, mxlen=None, mxwlen=None):
        super(ElmoPreprocessorCreator, self).__init__(indices, lchars, upchars_lut, task, token_key, extra_feats,
                                                      mxlen=mxlen, mxwlen=mxwlen)

    def preproc_post(self, post_mappings):
        # Split the input string, assuming that whitespace is splitter
        # The client should perform any required tokenization for us and join on ' '

        raw_post = post_mappings[self.token_key]
        # raw_post = post_mappings
        mxlen = self.mxlen
        mxwlen = self.mxwlen

        nraw_post = self._reform_raw(raw_post, mxlen)

        preprocs = {}
        words, sentence_length = self._create_word_vectors_from_post(nraw_post, mxlen)
        mixed_case_words = self._create_word_vectors_from_post_mixed_case(nraw_post, mxlen)
        preprocs['word'] = words
        preprocs['mixed_case_word'] = mixed_case_words
        if 'char' in self.indices:
            chars, _ = self._create_char_vectors_from_post(nraw_post, mxlen, mxwlen)
            preprocs['char'] = chars
        return preprocs, sentence_length

    def _create_word_vectors_from_post_mixed_case(self, nraw_post, mxlen):
        # vocab has only lowercase words
        word_tokens = tf.string_split(tf.reshape(nraw_post, [-1]))

        word_indices = self.word2index.lookup(word_tokens)

        # Reshape them out to the proper length
        reshaped_words = tf.sparse_reshape(word_indices, shape=[-1])
        x = self._reshape_indices(reshaped_words, [mxlen])

        return x


class ElmoTaggerTensorFlowExporter(TensorFlowExporter):
    def __init__(self, task):
        super(ElmoTaggerTensorFlowExporter, self).__init__(task)

    def _run(self, sess, model_file, embeddings_set, output_dir, model_version, use_preproc=True):

        indices, vocabs = self._create_vocabs(model_file)

        self.assign_char_lookup()

        labels = self.load_labels(model_file)
        extra_features_required = []

        # Make the TF example, network input
        serialized_tf_example, tf_example = self._create_example(extra_features_required)
        raw_posts = tf_example[FIELD_NAME]

        mxlen, mxwlen = self._get_max_lens(model_file)

        preprocessor = ElmoPreprocessorCreator(
            indices, self.lchars, self.upchars_lut,
            self.task, FIELD_NAME, extra_features_required, mxlen, mxwlen
        )

        types = {k: tf.int64 for k in indices.keys()}
        types.update({'mixed_case_word': tf.int64})
        # Run for each post

        preprocessed, lengths = tf.map_fn(preprocessor.preproc_post, tf_example,
                                          dtype=(types, tf.int32),
                                          back_prop=False)
        embeddings = self._initialize_embeddings_map(vocabs, embeddings_set)

        model_params = self.task.config_params["model"]
        model_params["x"] = preprocessed['mixed_case_word']
        model_params["xch"] = preprocessed['char']
        model_params["x_lc"] = preprocessed['word']

        model_params["lengths"] = lengths
        model_params["pkeep"] = 1
        model_params["sess"] = sess
        model_params["maxs"] = mxlen
        model_params["maxw"] = mxwlen
        model_params['span_type'] = self.task.config_params['train'].get('span_type')
        print(model_params)
        model = baseline.tf.tagger.create_model(labels, embeddings, **model_params)
        model.create_loss()

        softmax_output = tf.nn.softmax(model.probs)
        values, indices = tf.nn.top_k(softmax_output, 1)

        start_np = np.full((1, 1, len(labels)), -1e4, dtype=np.float32)
        start_np[:, 0, labels['<GO>']] = 0
        start = tf.constant(start_np)
        model.probs = tf.concat([start, model.probs], 1)

        if model.crf is True:
            indices, _ = tf.contrib.crf.crf_decode(model.probs, model.A, tf.constant([mxlen + 1]))## We are assuming the batchsz is 1 here
            indices = indices[:, 1:]

        list_of_labels = [''] * len(labels)
        for label, idval in labels.items():
            list_of_labels[idval] = label

        class_tensor = tf.constant(list_of_labels)
        table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
        classes = table.lookup(tf.to_int64(indices))
        self.restore_checkpoint(sess, model_file)

        sig_input = SignatureInput(None, raw_posts)
        sig_output = SignatureOutput(classes, values)

        return sig_input, sig_output, "tag_text"


def create_exporter(task, exporter_type):
    return ElmoTaggerTensorFlowExporter(task)

import json
import tensorflow as tf
import os
from baseline.utils import Offsets, export
from mead.preprocessors import Preprocessor, register_preprocessor

__all__ = []
exporter = export(__all__)


@exporter
class TensorFlowPreprocessor(Preprocessor):
    """
    Generic class for creating vectorizers using tensorflow ops, to be used for exporting models when the service gets
    a string instead of a vectorized input.
    """

    def __init__(self, model_base_dir, pid, feature, vectorizer, **kwargs):
        super(TensorFlowPreprocessor, self).__init__(model_base_dir, pid, feature, vectorizer, **kwargs)
        self.index, self.vocab = self.create_vocab()

    def create_vocab(self):
        """
        :model_file the path-like object to the model and model name.
        :vocab_suffixes the list of vocab types. e.g. 'word', 'char', 'ner'.
        """
        index, vocab = None, None
        feature_vocab_file = os.path.join(self.model_base_dir, "vocabs-{}-{}.json".format(self.feature, self.pid))
        if os.path.exists(feature_vocab_file):
            index, vocab = self._read_vocab(feature_vocab_file, self.feature)
        return index, vocab

    # def create_vectorizers(self):
    #     """
    #     :model_file the path-like object to the model and model name.
    #     :vocab_suffixes the list of vocab types. e.g. 'word', 'char', 'ner'.
    #     """
    #     import pickle
    #     return pickle.load(open(os.path.join(self.model_base_dir, "vectorizers-{}.pkl".format(self.pid)), "rb"))

    def preproc(self, tf_example):
        """
        Create a preprocessor chain inside of the tensorflow graph.
        """
        pass

    @staticmethod
    def resize_sen(raw, mxlen):
        """
        Splits and rejoins a string to ensure that tokens meet
        the required max len.
        """
        raw_tokens = tf.string_split(tf.reshape(raw, [-1])).values
        # sentence length <= mxlen
        raw_post = tf.reduce_join(raw_tokens[mxlen], separator=" ")
        return raw_post

    @staticmethod
    def lowercase(raw_post):
        upchars = tf.constant([chr(i) for i in range(65, 91)])
        lchars = tf.constant([chr(i) for i in range(97, 123)])
        upchars_lut = tf.contrib.lookup.index_table_from_tensor(mapping=upchars, num_oov_buckets=1, default_value=-1)

        split_chars = tf.string_split(tf.reshape(raw_post, [-1]), delimiter="").values
        upchar_inds = upchars_lut.lookup(split_chars)
        return tf.reduce_join(tf.map_fn(lambda x: tf.cond(x[0] > 25,
                                                          lambda: x[1],
                                                          lambda: lchars[x[0]]),
                                        (upchar_inds, split_chars), dtype=tf.string))

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

    @staticmethod
    def reshape_indices(indices, shape):
        reshaped = tf.sparse_reset_shape(indices, new_shape=shape)
        # Now convert to a dense representation
        x = tf.sparse_tensor_to_dense(reshaped)
        x = tf.contrib.framework.with_shape(shape, x)
        return x


@exporter
@register_preprocessor(name='token1d')
class Token1DPreprocessor(TensorFlowPreprocessor):

    def __init__(self, model_base_dir, pid, feature, vectorizer, **kwargs):
        super(Token1DPreprocessor, self).__init__(model_base_dir, pid, feature, vectorizer, **kwargs)
        self.mxlen = self.vectorizer.mxlen

    def create_word_vectors_from_post(self, raw_post, lowercase=True):
        # vocab has only lowercase words
        word2index = self.index
        if lowercase:
            raw_post = self.lowercase(raw_post)
        word_tokens = tf.string_split(tf.reshape(raw_post, [-1]))
        word_indices = word2index.lookup(word_tokens)
        # Reshape them out to the proper length
        reshaped_words = tf.sparse_reshape(word_indices, shape=[-1])
        return self.reshape_indices(reshaped_words, [self.mxlen])

    def preproc(self, post_mappings):
        # Split the input string, assuming that whitespace is splitter
        # The client should perform any required tokenization for us and join on ' '
        raw_post = post_mappings[self.feature]
        raw_post = self.resize_sen(raw_post, self.mxlen)
        return self.create_word_vectors_from_post(raw_post)


@exporter
@register_preprocessor(name='char2d')
class Char2DPreprocessor(TensorFlowPreprocessor):
    def __init__(self, model_base_dir, pid, feature, vectorizer, **kwargs):
        super(Char2DPreprocessor, self).__init__(model_base_dir, pid, feature, vectorizer, **kwargs)
        self.mxlen = self.vectorizer.mxlen
        self.mxwlen = self.vectorizer.mxwlen

    def create_char_vectors_from_post(self, raw_post, lowercase=False):
        char2index = self.index
        if lowercase:
            raw_post = self.lowercase(raw_post)
        raw_post = tf.string_split(tf.reshape(raw_post, [-1]))
        culled_word_token_vals = tf.substr(raw_post.values, 0, self.mxwlen)
        char_tokens = tf.string_split(culled_word_token_vals, delimiter='')
        char_indices = char2index.lookup(char_tokens)
        return self.reshape_indices(char_indices, [self.mxlen, self.mxwlen])

    def preproc(self, post_mappings):
        raw_post = post_mappings[self.feature]
        raw_post = self.resize_sen(raw_post, self.mxlen)
        return self.create_char_vectors_from_post(raw_post)

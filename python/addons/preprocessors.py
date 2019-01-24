import json
import tensorflow as tf
import os
from baseline.utils import Offsets

class PreprocessorCreator(object):
    """
    Generic class for creating using tensorflow ops
    """

    def __init__(self, model_base_dir, pid, features, **kwargs):
        self.indices, self.vocabs = self.create_vocabs(model_base_dir, pid, features)
        self.vectorizers = self.create_vectorizers(model_base_dir, pid)
        self.FIELD_NAME = None

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
    def create_vectorizers(model_base_dir, pid):
        """
        :model_file the path-like object to the model and model name.
        :vocab_suffixes the list of vocab types. e.g. 'word', 'char', 'ner'.
        """
        import pickle
        return pickle.load(open(os.path.join(model_base_dir, "vectorizers-{}.pkl".format(pid)), "rb"))

    def _assign_char_lookup(self):
        upchars = tf.constant([chr(i) for i in range(65, 91)])
        self.lchars = tf.constant([chr(i) for i in range(97, 123)])
        self.upchars_lut = tf.contrib.lookup.index_table_from_tensor(mapping=upchars, num_oov_buckets=1, default_value=-1)

    def _create_example(self):
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {
            self.FIELD_NAME: tf.FixedLenFeature(shape=[], dtype=tf.string),
        }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        return tf_example

    def run(self):
        self._assign_char_lookup()
        tf_example = self._create_example()
        preprocessed = self.create_preprocessed_input(tf_example)
        return tf_example, preprocessed

    def create_preprocessed_input(self, tf_example):
        """
        Create a preprocessor chain inside of the tensorflow graph.
        """
        pass

    @staticmethod
    def reshape_indices(indices, shape):
        reshaped = tf.sparse_reset_shape(indices, new_shape=shape)
        # Now convert to a dense representation
        x = tf.sparse_tensor_to_dense(reshaped)
        x = tf.contrib.framework.with_shape(shape, x)
        return x


class Token1DPreprocessorCreator(PreprocessorCreator):

    def __init__(self, model_base_dir, pid, features, **kwargs):
        super(Token1DPreprocessorCreator, self).__init__(model_base_dir, pid, features, **kwargs)
        self.mxlen = self.vectorizers['word'].mxlen
        self.FIELD_NAME = 'tokens'

    def reform_raw(self, raw):
        """
        Splits and rejoins a string to ensure that tokens meet
        the required max len.
        """
        raw_tokens = tf.string_split(tf.reshape(raw, [-1])).values
        # sentence length <= mxlen
        raw_post = tf.reduce_join(raw_tokens[:self.mxlen], separator=" ")
        return raw_post

    def create_word_vectors_from_post(self, raw_post, lowercase=True):
        # vocab has only lowercase words
        word2index = self.indices['word']
        if lowercase:
            split_chars = tf.string_split(tf.reshape(raw_post, [-1]), delimiter="").values
            upchar_inds = self.upchars_lut.lookup(split_chars)
            raw_post = tf.reduce_join(tf.map_fn(lambda x: tf.cond(x[0] > 25,
                                                                     lambda: x[1],
                                                                     lambda: self.lchars[x[0]]),
                                                   (upchar_inds, split_chars), dtype=tf.string))
        word_tokens = tf.string_split(tf.reshape(raw_post, [-1]))
        word_indices = word2index.lookup(word_tokens)
        # Reshape them out to the proper length
        reshaped_words = tf.sparse_reshape(word_indices, shape=[-1])
        return self.reshape_indices(reshaped_words, [self.mxlen])

    def preproc(self, post_mappings):
        # Split the input string, assuming that whitespace is splitter
        # The client should perform any required tokenization for us and join on ' '

        raw_post = post_mappings[self.FIELD_NAME]
        raw_post = self.reform_raw(raw_post)
        return {'word': self.create_word_vectors_from_post(raw_post)}

    def create_preprocessed_input(self, tf_example):
        """
        Create a preprocessor chain inside of the tensorflow graph.
        """
        types = {'word': tf.int64}
        return tf.map_fn(
            self.preproc, tf_example,
            dtype=types, back_prop=False
        )


class Token2DPreprocessorCreator(Token1DPreprocessorCreator):

    def __init__(self, model_base_dir, pid, features, **kwargs):
        super(Token2DPreprocessorCreator, self).__init__(model_base_dir, pid, features, **kwargs)
        self.mxwlen = self.vectorizers['char'].mxwlen

    def create_char_vectors_from_post(self, raw_post):
        char2index = self.indices['char']
        unchanged_word_tokens = tf.string_split(tf.reshape(raw_post, [-1]))
        culled_word_token_vals = tf.substr(unchanged_word_tokens.values, 0, self.mxwlen)
        char_tokens = tf.string_split(culled_word_token_vals, delimiter='')
        char_indices = char2index.lookup(char_tokens)
        return self.reshape_indices(char_indices, [self.mxlen, self.mxwlen])

    def preproc(self, post_mappings):
        # Split the input string, assuming that whitespace is splitter
        # The client should perform any required tokenization for us and join on ' '

        raw_post = post_mappings[self.FIELD_NAME]
        raw_post = self.reform_raw(raw_post)
        return {
            'word': self.create_word_vectors_from_post(raw_post),
            'char': self.create_char_vectors_from_post(raw_post)
        }

    def create_preprocessed_input(self, tf_example):
        """
        Create a preprocessor chain inside of the tensorflow graph.
        """
        types = {'word': tf.int64, 'char': tf.int64}
        return tf.map_fn(
            self.preproc, tf_example,
            dtype=types, back_prop=False
        )

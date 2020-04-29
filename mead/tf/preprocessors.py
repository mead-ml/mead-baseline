import tensorflow as tf
from baseline.utils import exporter
from mead.preprocessors import Preprocessor, register_preprocessor

__all__ = []
export = exporter(__all__)


@export
class TensorFlowPreprocessor(Preprocessor):
    """
    Generic class for creating vectorizers using tensorflow ops, to be used for exporting models when the service gets
    a string instead of a vectorized input.
    """

    def __init__(self, feature, vectorizer, index, vocab, **kwargs):
        super(TensorFlowPreprocessor, self).__init__(feature, vectorizer, index, vocab, **kwargs)
        self.FIELD_NAME = kwargs.get('FIELD_NAME', 'tokens')
        self.upchars = tf.constant([chr(i) for i in range(65, 91)])
        self.lchars = tf.constant([chr(i) for i in range(97, 123)])
        self.upchars_lut = tf.contrib.lookup.index_table_from_tensor(mapping=self.upchars, num_oov_buckets=1, default_value=-1)

    def lowercase(self, raw_post):
        split_chars = tf.string_split(tf.reshape(raw_post, [-1]), delimiter="").values
        upchar_inds = self.upchars_lut.lookup(split_chars)
        return tf.reduce_join(tf.map_fn(lambda x: tf.cond(x[0] > 25,
                                                          lambda: x[1],
                                                          lambda: self.lchars[x[0]]),
                                        (upchar_inds, split_chars), dtype=tf.string))

    def preproc(self, tf_example, mxlen):
        """
        Create a preprocessor chain inside of the tensorflow graph.
        """
        pass

    def resize_sen(self, raw, mxlen):
        """
        Splits and rejoins a string to ensure that tokens meet
        the required max len.
        """
        raw_tokens = tf.string_split(tf.reshape(raw, [-1])).values
        # sentence length > mxlen
        raw_post = tf.reduce_join(raw_tokens[:mxlen], separator=" ")
        return raw_post

    @staticmethod
    def reshape_indices(indices, shape):
        reshaped = tf.sparse_reset_shape(indices, new_shape=shape)
        # Now convert to a dense representation
        x = tf.sparse_tensor_to_dense(reshaped)
        return x


@export
@register_preprocessor(name='token1d')
class Token1DPreprocessor(TensorFlowPreprocessor):

    def __init__(self, feature, vectorizer, index, vocab, **kwargs):
        super().__init__(feature, vectorizer, index, vocab, **kwargs)
        self.hard_mxlen = self.vectorizer.mxlen
        # 10000 is just a large max, six.MAXSIZE was causing issues
        self.hard_mxlen = self.hard_mxlen if self.hard_mxlen != -1 else 10000

        transform_fn = self.vectorizer.transform_fn.__name__
        if transform_fn not in ["identity_trans_fn", "lowercase"]:
            raise NotImplementedError("can not export arbitrary transform functions")
        self.do_lowercase = transform_fn == "lowercase"

    def create_word_vectors_from_post(self, raw_post, mxlen):
        # vocab has only lowercase words
        word2index = self.index
        if self.do_lowercase:
            raw_post = self.lowercase(raw_post)
        word_tokens = tf.string_split(tf.reshape(raw_post, [-1]))
        word_indices = word2index.lookup(word_tokens)
        # Reshape them out to the proper length
        reshaped_words = tf.sparse_reshape(word_indices, shape=[-1])
        return self.reshape_indices(reshaped_words, [mxlen])

    def preproc(self, post_mappings, mxlen):
        # Split the input string, assuming that whitespace is splitter
        # The client should perform any required tokenization for us and join on ' '
        mxlen = tf.minimum(mxlen, self.hard_mxlen)
        raw_post = post_mappings[self.FIELD_NAME]
        raw_post = self.resize_sen(raw_post, mxlen)
        return self.create_word_vectors_from_post(raw_post, mxlen)


@export
@register_preprocessor(name='char2d')
class Char2DPreprocessor(TensorFlowPreprocessor):
    def __init__(self, feature, vectorizer, index, vocab, **kwargs):
        super().__init__(feature, vectorizer, index, vocab, **kwargs)
        self.hard_mxlen = self.vectorizer.mxlen
        # 10000 is just a large max, six.MAXSIZE was causing issues
        self.hard_mxlen = self.hard_mxlen if self.hard_mxlen != -1 else 10000
        self.mxwlen = self.vectorizer.mxwlen
        transform_fn = self.vectorizer.transform_fn.__name__
        if transform_fn not in ["identity_trans_fn", "lowercase"]:
            raise NotImplementedError("can not export arbitrary transform functions")
        self.do_lowercase = transform_fn == "lowercase"

    def create_char_vectors_from_post(self, raw_post, mxlen):
        char2index = self.index
        if self.do_lowercase:
            raw_post = self.lowercase(raw_post)
        raw_post = tf.string_split(tf.reshape(raw_post, [-1]))
        culled_word_token_vals = tf.substr(raw_post.values, 0, self.mxwlen)
        char_tokens = tf.string_split(culled_word_token_vals, delimiter='')
        char_indices = char2index.lookup(char_tokens)
        return self.reshape_indices(char_indices, [mxlen, self.mxwlen])

    def preproc(self, post_mappings, mxlen):
        mxlen = tf.minimum(mxlen, self.hard_mxlen)
        raw_post = post_mappings[self.FIELD_NAME]
        raw_post = self.resize_sen(raw_post, mxlen)
        return self.create_char_vectors_from_post(raw_post, mxlen)


@export
@register_preprocessor(name='dict1d')
class Dict1DPreprocessor(Token1DPreprocessor):
    def __init__(self, feature, vectorizer, index, vocab, **kwargs):
        super(Dict1DPreprocessor, self).__init__(feature, vectorizer, index, vocab, **kwargs)


@export
@register_preprocessor(name='dict2d')
class Dict2DPreprocessor(Char2DPreprocessor):
    def __init__(self, feature, vectorizer, index, vocab, **kwargs):
        super(Dict2DPreprocessor, self).__init__(feature, vectorizer, index, vocab, **kwargs)

import logging
from baseline.embeddings import register_embeddings
from eight_mile.tf.embeddings import *
import tensorflow as tf


logger = logging.getLogger('baseline')


class TensorFlowEmbeddingsMixin(tf.keras.layers.Layer):
    """This provides a base for TensorFlow embeddings sub-graphs that includes the placeholders

    """
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        """Constructor
        """
        super().__init__(trainable=trainable, name=name, dtype=dtype)
        self._record_state(**kwargs)

    def detached_ref(self):
        """This will detach any attached input and reference the same sub-graph otherwise

        TODO: this should not longer be required and can be removed

        :return:
        """
        if getattr(self, '_weights', None) is not None:
            return type(self)(name=self.name, weights=self._weights, **self._state)
        if hasattr(self, 'embed') and getattr(self.init_embed, '_weights') is not None:
            return type(self)(name=self.name, weights=self.init_embed._weights, **self._state)
        raise Exception('You must initialize `weights` in order to use this method')

    def call(self, x):
        if x is None:
            x = self.create_placeholder(self.name)
        self.x = x

        return super().encode(x)

    @classmethod
    def create_placeholder(cls, name):
        """Create a placeholder with name `name`

        :param name: (``str``) The name of the placeholder
        :return: The placeholder
        """
        pass

    @classmethod
    def create(cls, model, name, **kwargs):
        """Instantiate this sub-graph from the generalized representation from `baseline.w2v`

        :param name: The name of the embeddings
        :param model: The `baseline.w2v` model
        :param kwargs:
        :return:
        """
        kwargs.pop('dsz', None)
        return cls(name=name, vsz=model.vsz, dsz=model.dsz, weights=model.weights, **kwargs)

    def _record_state(self, **kwargs):
        w = kwargs.pop('weights', None)
        self._state = copy.deepcopy(kwargs)

    def save_md(self, target):
        """Save the metadata associated with this embedding as a JSON file

        :param target: The name of the output file
        :return:
        """
        write_json(self.get_config(), target)

    def get_config(self):
        #config = super(TensorFlowEmbeddings, self).get_config()
        config = {}
        config['dsz'] = int(self.get_dsz())
        config['vsz'] = int(self.get_vsz())
        config['module'] = self.__class__.__module__
        config['class'] = self.__class__.__name__
        config.update(self._state)
        return config


@register_embeddings(name='default')
class LookupTableEmbeddingsModel(LookupTableEmbeddings, TensorFlowEmbeddingsMixin):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None], name=name)


@register_embeddings(name='char-conv')
class CharConvEmbeddingsModel(CharConvEmbeddings, TensorFlowEmbeddingsMixin):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None, None], name=name)


@register_embeddings(name='char-transformer')
class CharTransformerModel(CharTransformerEmbeddings, TensorFlowEmbeddingsMixin):
    pass


@register_embeddings(name='char-lstm')
class CharLSTMEmbeddingsModel(CharLSTMEmbeddings, TensorFlowEmbeddingsMixin):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None, None], name=name)


@register_embeddings(name='positional')
class PositionalLookupTableEmbeddingsModel(PositionalLookupTableEmbeddings, TensorFlowEmbeddingsMixin):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None], name=name)


@register_embeddings(name='learned-positional')
class LearnedPositionalLookupTableEmbeddingsModel(LearnedPositionalLookupTableEmbeddings, TensorFlowEmbeddingsMixin):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None], name=name)


@register_embeddings(name='learned-positional-w-bias')
class LearnedPositionalLookupTableEmbeddingsWithBiasModel(LearnedPositionalLookupTableEmbeddingsWithBias, TensorFlowEmbeddingsMixin):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None], name=name)


@register_embeddings(name='positional-char-conv')
class PositionalCharConvEmbeddingsModel(PositionalCharConvEmbeddings, TensorFlowEmbeddingsMixin):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None, None], name=name)


@register_embeddings(name='learned-positional-char-conv')
class PositionalCharConvEmbeddingsModel(LearnedPositionalCharConvEmbeddings, TensorFlowEmbeddingsMixin):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None, None], name=name)


@register_embeddings(name='positional-char-lstm')
class PositionalCharLSTMEmbeddingsModel(PositionalCharLSTMEmbeddings, TensorFlowEmbeddingsMixin):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None, None], name=name)


@register_embeddings(name='learned-positional-char-lstm')
class LearnedPositionalCharLSTMEmbeddingsModel(LearnedPositionalCharLSTMEmbeddings, TensorFlowEmbeddingsMixin):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None, None], name=name)


import logging
from eight_mile.embeddings import register_embeddings
from eight_mile.tf.embeddings import *
import tensorflow as tf


logger = logging.getLogger('baseline')


class TensorFlowEmbeddingsModel(tf.keras.Model):
    """This provides a base for TensorFlow embeddings sub-graphs that includes the placeholders

    """
    def __init__(self, name=None, **kwargs):
        """Constructor
        """
        super().__init__()
        self._record_state(**kwargs)
        self._name = name
        self.embedding_layer = None

    def detached_ref(self):
        """This will detach any attached input and reference the same sub-graph otherwise

        :return:
        """
        if getattr(self.embedding_layer, '_weights', None) is not None:
            return type(self)(self._name, weights=self.embedding_layer._weights, **self._state)
        if hasattr(self.embedding_layer, 'embed') and getattr(self.embedding_layer.embed, '_weights') is not None:
            return type(self)(self._name, weights=self.embedding_layer.embed._weights, **self._state)
        raise Exception('You must initialize `weights` in order to use this method')

    def get_dsz(self):
        """Get the number of output dimension of this operation

        :return:
        """
        return self.embedding_layer.get_dsz()

    def get_weights(self):
        return self.embedding_layer.get_weights()

    @property
    def output_dim(self):
        return self.embedding_layer.output_dim

    def get_vsz(self):
        """Get the number of words (including <PAD>) in the vocabulary

        :return:
        """
        return self.embedding_layer.get_vsz()

    def encode(self, x):
        """This instantiates the sub-graph for this object and returns the output node

        :return:
        """
        if x is None:
            x = self.create_placeholder(self._name)
        self.x = x
        return self.embedding_layer(x)

    def call(self, x):
        return self.encode(x)

    def get_feed_dict(self):
        """Return a feed dict that is needed to initialize this embeddings."""
        return self.embedding_layer.get_feed_dict()

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
        # If we think we are going to hit the 2GB limit swap out the LUT
        # embeddings to use the placeholder trick to get around it.
        # if cls is LookupTableEmbeddingsModel and model.vsz * model.dsz * FLOAT32 > GB2:
        #     cls = LargeLookupTableEmbeddingsModel
        #     logger.warning("Embedding %s seems to be larger than 2GB", name)
        return cls(name, vsz=model.vsz, dsz=model.dsz, weights=model.weights, **kwargs)

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
class LookupTableEmbeddingsModel(TensorFlowEmbeddingsModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding_layer = LookupTableEmbeddings(name=self._name, **kwargs)

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None], name=name)


@register_embeddings(name='char-conv')
class CharConvEmbeddingsModel(TensorFlowEmbeddingsModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding_layer = CharConvEmbeddings(name=self._name, **kwargs)

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)


@register_embeddings(name='char-lstm')
class CharLSTMEmbeddingsModel(TensorFlowEmbeddingsModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding_layer = CharLSTMEmbeddings(name=self._name, **kwargs)

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)


@register_embeddings(name='positional')
class PositionalLookupTableEmbeddingsModel(TensorFlowEmbeddingsModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding_layer = PositionalLookupTableEmbeddings(name=self._name, **kwargs)

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None], name=name)


@register_embeddings(name='learned-positional')
class LearnedPositionalLookupTableEmbeddingsModel(TensorFlowEmbeddingsModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding_layer = LearnedPositionalLookupTableEmbeddings(name=self._name, **kwargs)

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None], name=name)


@register_embeddings(name='positional-char-conv')
class PositionalCharConvEmbeddingsModel(TensorFlowEmbeddingsModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding_layer = PositionalCharConvEmbeddings(name=self._name, **kwargs)

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)


@register_embeddings(name='learned-positional-char-conv')
class PositionalCharConvEmbeddingsModel(TensorFlowEmbeddingsModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding_layer = LearnedPositionalCharConvEmbeddings(name=self._name, **kwargs)

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)


@register_embeddings(name='positional-char-lstm')
class PositionalCharLSTMEmbeddingsModel(TensorFlowEmbeddingsModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding_layer = PositionalCharLSTMEmbeddings(name=self._name, **kwargs)

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)


@register_embeddings(name='learned-positional-char-lstm')
class LearnedPositionalCharLSTMEmbeddingsModel(TensorFlowEmbeddingsModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding_layer = LearnedPositionalCharLSTMEmbeddings(name=self._name, **kwargs)

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)


# @register_embeddings(name="large-lut")
# class LargeLookupTableEmbeddingsModel(TensorFlowEmbeddingsModel):
#     def __init__(self, name=None, **kwargs):
#         super().__init__(name, **kwargs)
#         self.embedding_layer = LargeLookupTableEmbeddings(name=self._name, **kwargs)

#     @classmethod
#     def create_placeholder(cls, name):
#         return tf.placeholder(tf.int32, [None, None], name=name)

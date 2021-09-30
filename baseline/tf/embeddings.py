import logging
from baseline.embeddings import register_embeddings, create_embeddings
from eight_mile.tf.embeddings import *
from eight_mile.tf.serialize import load_tlm_npz
from eight_mile.utils import read_json
from baseline.vectorizers import load_bert_vocab
from baseline.utils import MEAD_HUB_MODULES
import tensorflow as tf


logger = logging.getLogger('baseline')


class TensorFlowEmbeddingsMixin(tf.keras.layers.Layer):
    """This provides a base for TensorFlow embeddings sub-graphs that includes the placeholders

    """
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        """Constructor
        """
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
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

    def call(self, *x):

        if x[0] is None:
            x[0] = self.create_placeholder(self.name)
        self.x = x[0]

        return super().encode(*x)

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
        config['mead_hub_modules'] = MEAD_HUB_MODULES
        config.update(self._state)
        return config


@register_embeddings(name='default')
class LookupTableEmbeddingsModel(TensorFlowEmbeddingsMixin, LookupTableEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None], name=name)


@register_embeddings(name='char-conv')
class CharConvEmbeddingsModel(TensorFlowEmbeddingsMixin, CharConvEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None, None], name=name)


@register_embeddings(name='char-transformer')
class CharTransformerModel(TensorFlowEmbeddingsMixin, CharTransformerEmbeddings):
    pass


@register_embeddings(name='char-lstm')
class CharLSTMEmbeddingsModel(TensorFlowEmbeddingsMixin, CharLSTMEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None, None], name=name)


@register_embeddings(name='positional')
class PositionalLookupTableEmbeddingsModel(TensorFlowEmbeddingsMixin, PositionalLookupTableEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None], name=name)


@register_embeddings(name='learned-positional')
class LearnedPositionalLookupTableEmbeddingsModel(TensorFlowEmbeddingsMixin, LearnedPositionalLookupTableEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None], name=name)


@register_embeddings(name='learned-positional-w-bias')
class LearnedPositionalLookupTableEmbeddingsWithBiasModel(TensorFlowEmbeddingsMixin, LearnedPositionalLookupTableEmbeddingsWithBias):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None], name=name)


@register_embeddings(name='positional-char-conv')
class PositionalCharConvEmbeddingsModel(TensorFlowEmbeddingsMixin, PositionalCharConvEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None, None], name=name)


@register_embeddings(name='learned-positional-char-conv')
class PositionalCharConvEmbeddingsModel(TensorFlowEmbeddingsMixin, LearnedPositionalCharConvEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None, None], name=name)


@register_embeddings(name='positional-char-lstm')
class PositionalCharLSTMEmbeddingsModel(TensorFlowEmbeddingsMixin, PositionalCharLSTMEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None, None], name=name)


@register_embeddings(name='learned-positional-char-lstm')
class LearnedPositionalCharLSTMEmbeddingsModel(TensorFlowEmbeddingsMixin, LearnedPositionalCharLSTMEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None, None], name=name)


class TransformerLMEmbeddings(TensorFlowEmbeddings):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name)
        # You dont actually have to pass this if you are using the `load_bert_vocab` call from your
        # tokenizer.  In this case, a singleton variable will contain the vocab and it will be returned
        # by `load_bert_vocab`
        # If you trained your model with MEAD/Baseline, you will have a `*.json` file which would want to
        # reference here
        vocab_file = kwargs.get('vocab_file')
        if vocab_file and vocab_file.endswith('.json'):
            self.vocab = read_json(vocab_file)
        else:
            self.vocab = load_bert_vocab(kwargs.get('vocab_file'))

        # When we reload, allows skipping restoration of these embeddings
        # If the embedding wasnt trained with token types, this allows us to add them later
        self.skippable = set(listify(kwargs.get('skip_restore_embeddings', [])))

        self.cls_index = self.vocab.get('[CLS]', self.vocab.get('<s>'))
        self.vsz = max(self.vocab.values()) + 1
        self.d_model = int(kwargs.get('dsz', kwargs.get('d_model', 768)))
        self.init_embed(**kwargs)
        self.proj_to_dsz = tf.keras.layers.Dense(self.dsz, self.d_model) if self.dsz != self.d_model else _identity
        self.init_transformer(**kwargs)

    @property
    def dsz(self):
        return self.embeddings.output_dim

    def embed(self, input, token_type):
        return self.embeddings({'x': input, 'tt': token_type})

    def init_embed(self, **kwargs):
        # If you are using BERT, you probably want to use either
        # `learned-positional` with a token type feature
        # or `learned-positional-w-bias` if you dont care about the token type
        embed_type = kwargs.get('word_embed_type', 'learned-positional')
        x_embedding = create_embeddings(vsz=self.vsz, dsz=self.d_model, embed_type=embed_type, name='x')

        embeddings = {'x': x_embedding}
        # This is for BERT support when we are using 2 features
        token_type_vsz = kwargs.get('token_type_vsz')
        if token_type_vsz:
            tt_unif = float(kwargs.get('token_type_embed_unif', 0.001))
            tt_embedding = LookupTableEmbeddings(vsz=token_type_vsz, dsz=self.d_model, name='tt', padding_idx=None,
                                                 unif=tt_unif)
            embeddings['tt'] = tt_embedding
        # For bert, make sure this is `sum-layer-norm`
        reduction = kwargs.get('embeddings_reduction', kwargs.get('reduction', 'sum'))
        embeddings_dropout = kwargs.get('embeddings_dropout', 0.1)
        self.embeddings = EmbeddingsStack(embeddings, dropout_rate=embeddings_dropout, reduction=reduction)

    def init_transformer(self, **kwargs):
        num_layers = int(kwargs.get('layers', 12))
        num_heads = int(kwargs.get('num_heads', 12))
        pdrop = kwargs.get('dropout', 0.1)
        d_ff = int(kwargs.get('d_ff', 3072))
        d_k = kwargs.get('d_k')
        rpr_k = kwargs.get('rpr_k')
        layer_norms_after = kwargs.get('layer_norms_after', False)
        layer_norm_eps = kwargs.get('layer_norm_eps', 1e-12)
        activation = kwargs.get('activation', 'gelu')
        windowed_ra = kwargs.get('windowed_ra', False)
        rpr_value_on = kwargs.get('rpr_value_on', True)
        self.transformer = TransformerEncoderStack(num_heads, d_model=self.d_model, pdrop=pdrop, scale=True,
                                                   layers=num_layers, d_ff=d_ff, rpr_k=rpr_k, d_k=d_k,
                                                   activation=activation, layer_norms_after=layer_norms_after,
                                                   layer_norm_eps=layer_norm_eps, windowed_ra=windowed_ra,
                                                   rpr_value_on=rpr_value_on)
        self.mlm = kwargs.get('mlm', True)
        self.finetune = kwargs.get('finetune', True)

    def encode(self, x, token_type=None):
        # the following line masks out the attention to padding tokens
        input_mask = tf.expand_dims(tf.expand_dims(tf.cast(tf.not_equal(x, 0), tf.float32), 1), 1)
        # A causal LM should have a subsequent mask; and a masked LM should have no mask
        if not self.mlm:
            input_mask = tf.multiply(input_mask, subsequent_mask(x.shape[1]))
        embedding = self.embed(x, token_type)
        embedding = self.proj_to_dsz(embedding)
        transformer_out = self.transformer((embedding, input_mask))
        z = self.get_output(x, transformer_out)
        return z

    def get_output(self, inputs, z):
        return tf.stop_gradient(z) if self.finetune is False else z

    def get_vocab(self):
        return self.vocab

    def get_vsz(self):
        return self.vsz

    def get_dsz(self):
        return self.d_model

    @classmethod
    def load(cls, embeddings, **kwargs):
        c = cls("tlm-words-embed", **kwargs)

        # pass random data through to initialize the graph
        B = 1
        T = 8
        data_sample = tf.ones([B, T], dtype=tf.int32)
        token_type_vsz = kwargs.get('token_type_vsz')
        if token_type_vsz:
            _ = c(data_sample, data_sample)
        else:
        #data_sample_tt = tf.zeros([B, T], dtype=tf.int32)
            _ = c(data_sample)

        keys_to_restore = set(list(c.embeddings.keys()))
        filtered_keys = keys_to_restore.difference(c.skippable)
        if not keys_to_restore:
            raise Exception("No keys to restore!")
        if len(filtered_keys) < len(keys_to_restore):
            logger.warning("Restoring only key [%s]", ' '.join(filtered_keys))
        load_tlm_npz(c, embeddings, filtered_keys)
        return c

    #def detached_ref(self):
    #    return self


class TransformerLMPooledEmbeddings(TransformerLMEmbeddings):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

        pooling = kwargs.get('pooling', 'cls')
        if pooling == 'max':
            self.pooling_op = _max_pool
        elif pooling == 'mean':
            self.pooling_op = _mean_pool
        else:
            self.pooling_op = self._cls_pool

    def _cls_pool(self, inputs, tensor):
        pooled = tensor[tf.equal(inputs, self.cls_index)]
        return pooled

    def get_output(self, inputs, z):
        return self.pooling_op(inputs, z)



class TransformerLMPooled2DEmbeddings(TransformerLMPooledEmbeddings):

    def encode(self, xch):
        """Encode 3D data

        When the TLM rehydrates, it uses a (B, T) shape, which would cause an exception on 3D data, so
        we do a try block to allow it to hydrate first

        :param xch:
        :return:
        """
        try:
            _0, _1, W = get_shape_as_list(xch)
            xch = tf.reshape(xch, [-1, W])
            pooled = super().encode(xch)
            pooled = tf.reshape(pooled, [_0, _1, self.get_dsz()])
        except:
            pooled = super().encode(xch)

        return pooled


@register_embeddings(name='tlm-words-embed')
class TransformerLMEmbeddingsModel(TensorFlowEmbeddingsMixin, TransformerLMEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None], name=name)


@register_embeddings(name='tlm-words-embed-pooled')
class TransformerLMPooledEmbeddingsModel(TensorFlowEmbeddingsMixin, TransformerLMPooledEmbeddings):
    pass


@register_embeddings(name='tlm-words-embed-pooled2d')
class TransformerLMPooled2DEmbeddingsModel(TensorFlowEmbeddingsMixin, TransformerLMPooled2DEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None, None], name=name)


def _identity(x):
    return x

def _mean_pool(inputs, embeddings):
    mask = tf.not_equal(inputs, 0)
    seq_lengths = tf.reduce_sum(tf.cast(mask, tf.int8), axis=1, keepdims=True)
    embeddings = tf.where(tf.expand_dims(mask, -1), embeddings, 0.)
    return tf.reduce_sum(embeddings, 1, False) / tf.cast(seq_lengths, embeddings.dtype)


def _max_pool(inputs, embeddings):
    mask = tf.not_equal(inputs, 0)
    embeddings = tf.where(tf.expand_dims(mask, -1), embeddings, 0.)
    return tf.reduce_max(embeddings, 1, False)

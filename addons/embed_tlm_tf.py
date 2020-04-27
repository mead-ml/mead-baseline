import numpy as np

import tensorflow as tf
from eight_mile.tf.serialize import load_tlm_npz
from eight_mile.tf.layers import TransformerEncoderStack, EmbeddingsStack, subsequent_mask
from baseline.embeddings import register_embeddings, create_embeddings
from eight_mile.utils import Offsets, read_json
from baseline.vectorizers import load_bert_vocab
from eight_mile.tf.embeddings import TensorFlowEmbeddings, LookupTableEmbeddings, PositionalLookupTableEmbeddings, LearnedPositionalLookupTableEmbeddings
from baseline.tf.embeddings import TensorFlowEmbeddingsModel


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
            self.vocab = read_json(kwargs.get('vocab_file'))
        else:
            self.vocab = load_bert_vocab(kwargs.get('vocab_file'))

        self.cls_index = self.vocab['[CLS]']
        self.vsz = max(self.vocab.values()) + 1
        self.d_model = int(kwargs.get('dsz', kwargs.get('d_model', 410)))
        self.init_embed(**kwargs)
        self.proj_to_dsz = tf.keras.layers.Dense(self.dsz, self.d_model) if self.dsz != self.d_model else _identity
        self.init_transformer(**kwargs)

    @property
    def dsz(self):
        return self.embeddings.output_dim

    def embed(self, input):
        return self.embeddings(input)

    def init_embed(self, **kwargs):
        # If you are using BERT, you probably want to use either
        # `learned-positional` with a token type feature
        # or `learned-positional-w-bias` if you dont care about the token type
        embed_type = kwargs.get('word_embed_type', 'learned-positional')
        x_embedding = create_embeddings(vsz=self.vsz, dsz=self.d_model, embed_type=embed_type)

        embeddings = {'x': x_embedding}
        # This is for BERT support when we are using 2 features
        token_type_vsz = kwargs.get('token_type_vsz')
        if token_type_vsz:
            tt_embedding = LookupTableEmbeddings(vsz=token_type_vsz, dsz=self.dsz)
            embeddings['tt'] = tt_embedding
        # For bert, make sure this is `sum-layer-norm`
        reduction = kwargs.get('embeddings_reduction', kwargs.get('reduction'))
        embeddings_dropout = kwargs.get('embeddings_dropout', 0.1)
        self.embeddings = EmbeddingsStack(embeddings, dropout_rate=embeddings_dropout, reduction=reduction)
        return self.embeddings.output_dim

    def init_transformer(self, **kwargs):
        num_layers = int(kwargs.get('layers', 18))
        num_heads = int(kwargs.get('num_heads', 10))
        pdrop = kwargs.get('dropout', 0.1)
        d_ff = int(kwargs.get('d_ff', 2100))
        d_k = kwargs.get('d_k')
        rpr_k = kwargs.get('rpr_k')
        layer_norms_after = kwargs.get('layer_norms_after', False)
        layer_norm_eps = kwargs.get('layer_norm_eps', 1e-12)
        self.transformer = TransformerEncoderStack(num_heads, d_model=self.d_model, pdrop=pdrop, scale=True,
                                                   layers=num_layers, d_ff=d_ff, rpr_k=rpr_k, d_k=d_k,
                                                   layer_norms_after=layer_norms_after, layer_norm_eps=layer_norm_eps)
        self.mlm = kwargs.get('mlm', False)
        self.finetune = kwargs.get('finetune', True)

    def _model_mask(self, nctx):
        """This function creates the mask that controls which token to be attended to depending on the model. A causal
        LM should have a subsequent mask; and a masked LM should have no mask."""
        if self.mlm:
            return tf.fill([1, 1, nctx, nctx], 1.)
        else:
            return subsequent_mask(nctx)

    def encode(self, x):
        # the following line masks out the attention to padding tokens
        input_mask = tf.expand_dims(tf.expand_dims(tf.cast(tf.not_equal(x, 0), tf.float32), 1), 1)
        # the following line builds mask depending on whether it is a causal lm or masked lm
        input_mask = tf.multiply(input_mask, self._model_mask(x.shape[1]))
        embedding = self.embed(x)
        embedding = self.proj_to_dsz(embedding)
        transformer_out = self.transformer((embedding, input_mask))
        z = self.get_output(x, transformer_out)
        return z

    def get_output(self, inputs, z):
        return tf.stop_gradient(z)

    def get_vocab(self):
        return self.vocab

    def get_vsz(self):
        return self.vsz

    def get_dsz(self):
        return self.d_model


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


@register_embeddings(name='tlm-words-embed')
class TransformerLMEmbeddingsModel(TensorFlowEmbeddingsModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding_layer = TransformerLMEmbeddings(name=self._name, **kwargs)

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None], name=name)

    @classmethod
    def load(cls, embeddings, **kwargs):
        c = cls("tlm-words-embed", **kwargs)
        # pass random data through to initialize the graph
        B = 1
        T = 8
        data_sample = tf.ones([B, T], dtype=tf.int32)
        _ = c(data_sample)
        if embeddings.endswith('.npz'):
            load_tlm_npz(c.embedding_layer, embeddings)
        else:
            raise Exception("Can only load npz checkpoint to TF for now.")
        return c

    def get_vocab(self):
        return self.embedding_layer.get_vocab()

    def detached_ref(self):
        return self


@register_embeddings(name='tlm-words-embed-pooled')
class TransformerLMPooledEmbeddingsModel(TransformerLMEmbeddingsModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding_layer = TransformerLMPooledEmbeddings(name=self._name, **kwargs)


def _identity(x):
    return x


def _mean_pool(_, embeddings):
    return tf.reduce_mean(embeddings, 1, False)


def _max_pool(_, embeddings):
    return tf.reduce_max(embeddings, 1, False)




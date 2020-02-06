from collections import Counter

import tensorflow as tf
from eight_mile.tf.serialize import load_tlm_npz
from eight_mile.tf.layers import TransformerEncoderStack
from eight_mile.tf.layers import EmbeddingsStack, subsequent_mask
from eight_mile.embeddings import register_embeddings
from eight_mile.utils import Offsets, read_json
from baseline.vectorizers import register_vectorizer, AbstractVectorizer
from eight_mile.tf.embeddings import TensorFlowEmbeddings
from baseline.tf.embeddings import TensorFlowEmbeddingsModel, PositionalLookupTableEmbeddingsModel


class SavableFastBPE(object):
    def __init__(self, codes_path, vocab_path):
        from fastBPE import fastBPE
        self.codes = open(codes_path, 'rb').read()
        self.vocab = open(vocab_path, 'rb').read()
        self.bpe = fastBPE(codes_path, vocab_path)

    def __getstate__(self):
        return {'codes': self.codes, 'vocab': self.vocab}

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile() as codes, tempfile.NamedTemporaryFile() as vocab:
            codes.write(state['codes'])
            vocab.write(state['vocab'])
            self.bpe = fastBPE(codes.name, vocab.name)

    def apply(self, sentences):
        return self.bpe.apply(sentences)


@register_vectorizer(name='tlm-bpe')
class BPEVectorizer1D(AbstractVectorizer):
    """Define a Baseline Vectorizer for BPE using fastBPE (https://github.com/glample/fastBPE)
    If you use tokens=bpe, this vectorizer is used, and so then there is a
    dependency on fastBPE
    To use BPE, we assume that a Dictionary of codes and vocab was already created
    """
    def __init__(self, **kwargs):
        """Loads a BPE tokenizer"""
        super(BPEVectorizer1D, self).__init__(kwargs.get('transform_fn'))
        self.max_seen = 128
        self.model_file = kwargs.get('model_file')
        self.vocab_file = kwargs.get('vocab_file')
        self.tokenizer = SavableFastBPE(self.model_file, self.vocab_file)
        self.mxlen = kwargs.get('mxlen', -1)
        self.vocab = {k: i for i, k in enumerate(self.read_vocab(self.vocab_file))}

    def read_vocab(self, s):
        vocab = [] + Offsets.VALUES + ['[CLS]']
        with open(s, "r") as f:
            for line in f.readlines():
                token = line.split()[0].strip()
                vocab.append(token)
        return vocab

    def count(self, tokens):
        seen = 0
        counter = Counter()
        for tok in self.iterable(tokens):
            counter[tok] += 1
            seen += 1
        self.max_seen = max(self.max_seen, seen)
        return counter

    def iterable(self, tokens):
        for t in tokens:
            if t in Offsets.VALUES:
                yield t
            elif t == '<unk>':
                yield Offsets.VALUES[Offsets.UNK]
            elif t == '<eos>':
                yield Offsets.VALUES[Offsets.EOS]
            else:
                subwords = self.tokenizer.apply([t])[0].split()
                for x in subwords:
                    yield x
        yield '[CLS]'

    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            value = vocab.get(atom)
            if value is None:
                value = vocab[Offsets.VALUES[Offsets.UNK]]
            yield value

    def run(self, tokens, vocab):
        if self.mxlen < 0:
            self.mxlen = self.max_seen
        vec1d = np.zeros(self.mxlen, dtype=np.long)
        for i, atom in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= 1
                vec1d[i] = vocab.get('[CLS]')
                break
            vec1d[i] = atom
        valid_length = i + 1
        return vec1d, valid_length

    def get_dims(self):
        return self.mxlen,


class TransformerLMEmbeddings(TensorFlowEmbeddings):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name)
        self.vocab = read_json(kwargs.get('vocab_file'))
        self.cls_index = self.vocab['[CLS]']
        self.vsz = len(self.vocab)
        layers = int(kwargs.get('layers', 18))
        num_heads = int(kwargs.get('num_heads', 10))
        pdrop = kwargs.get('dropout', 0.1)
        self.d_model = int(kwargs.get('dsz', kwargs.get('d_model', 410)))
        d_ff = int(kwargs.get('d_ff', 2100))
        x_embedding = PositionalLookupTableEmbeddingsModel(name=self._name, vsz=self.vsz, dsz=self.d_model)
        self.dsz = self.init_embed({'x': x_embedding})
        self.proj_to_dsz = tf.keras.layers.Dense(self.d_model) if self.dsz != self.d_model else _identity
        self.transformer = TransformerEncoderStack(layers=layers, d_model=self.d_model, pdrop=pdrop,
                                                   num_heads=num_heads, d_ff=d_ff)
        self.mlm = kwargs.get('mlm', False)

    def embed(self, input):
        embedded = self.embeddings['x'](input)
        embedded_dropout = self.embed_dropout(embedded)
        if self.embeddings_proj:
            embedded_dropout = self.embeddings_proj(embedded_dropout)
        return embedded_dropout

    def init_embed(self, embeddings, **kwargs):
        pdrop = float(kwargs.get('dropout', 0.1))
        self.embed_dropout = tf.keras.layers.Dropout(pdrop)
        self.embeddings = EmbeddingsStack(embeddings)
        input_sz = 0
        for k, embedding in embeddings.items():
            input_sz += embedding.get_dsz()

        projsz = kwargs.get('projsz')
        if projsz:
            self.embeddings_proj = tf.keras.layers.Dense(projsz)
            print('Applying a transform from {} to {}'.format(input_sz, projsz))
            return projsz
        else:
            self.embeddings_proj = None
        return input_sz

    def _model_mask(self, nctx):
        """This function creates the mask that controls which token to be attended to depending on the model. A causal
        LM should have a subsequent mask; and a masked LM should have no mask."""
        if self.mlm:
            return tf.fill([1, 1, nctx, nctx], True)
        else:
            return subsequent_mask(nctx)

    def encode(self, x):
        # the following line masks out the attention to padding tokens
        input_mask = tf.expand_dims(tf.expand_dims(tf.not_equal(x, 0), 1), 1)
        # the following line builds mask depending on whether it is a causal lm or masked lm
        input_mask = tf.logical_and(input_mask, self._model_mask(x.shape[1]).type_as(input_mask))
        embedding = self.embed(x)
        embedding = self.proj_to_dsz(embedding)
        z = self.get_output(x, self.transformer((embedding, input_mask)))
        return z

    def get_output(self, inputs, z):
        return tf.stop_gradient(z)

    def get_vocab(self):
        return self.vocab

    def get_vsz(self):
        return self.vsz

    def get_dsz(self):
        return self.d_model


@register_embeddings(name='tlm-word-embed')
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
        if embeddings.endswith('.npz'):
            load_tlm_npz(c, embeddings)
        else:
            raise Exception("Can only load npz checkpoint to TF now.")
        return c


def _identity(x):
    return x


def _mean_pool(_, embeddings):
    return tf.reduce_mean(embeddings, 1, False)


def _max_pool(_, embeddings):
    return tf.reduce_max(embeddings, 1, False)


@register_embeddings(name='tlm-words-embed-pooled')
class TransformerLMPooledEmbeddingsModel(TransformerLMEmbeddingsModel):

    def __init__(self, name, **kwargs):
        super(TransformerLMPooledEmbeddingsModel, self).__init__(name=name, **kwargs)

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from eight_mile.pytorch.layers import TransformerEncoderStack, subsequent_mask
from eight_mile.pytorch.embeddings import PyTorchEmbeddings, PositionalLookupTableEmbeddings, LearnedPositionalLookupTableEmbeddings
from baseline.embeddings import register_embeddings, create_embeddings
from baseline.pytorch.embeddings import PyTorchEmbeddingsModel, BERTLookupTableEmbeddingsModel
from baseline.vectorizers import register_vectorizer, AbstractVectorizer, BPEVectorizer1D
from baseline.pytorch.torchy import *
from baseline.vectorizers import load_bert_vocab
from eight_mile.pytorch.serialize import load_tlm_transformers_bin
import torch.nn as nn


class BERTEmbeddings(PyTorchEmbeddings):
    """Support embeddings trained with the TransformerLanguageModel class

    This method supports either subword or word embeddings, not characters

    """
    def __init__(self, **kwargs):
        super().__init__()
        self.vocab = load_bert_vocab(kwargs.get('vocab_file'))
        self.cls_index = self.vocab['[CLS]']
        self.vsz = max(self.vocab.values()) + 1
        layers = int(kwargs.get('layers', 12))
        num_heads = int(kwargs.get('num_heads', 12))
        pdrop = kwargs.get('dropout', 0.1)
        self.d_model = int(kwargs.get('dsz', kwargs.get('d_model', 768)))
        d_ff = int(kwargs.get('d_ff', 3072))
        x_embedding = BERTLookupTableEmbeddingsModel(vsz=self.vsz, dsz=self.d_model, tok_type_vsz=2)
        self.dsz = self.init_embed({'x': x_embedding})
        self.proj_to_dsz = pytorch_linear(self.dsz, self.d_model) if self.dsz != self.d_model else _identity
        self.transformer = TransformerEncoderStack(num_heads, d_model=self.d_model, pdrop=pdrop, scale=True,
                                                   layers=layers, d_ff=d_ff, layer_norms_after=True)
        self.finetune = kwargs.get('finetune', True)

    def embed(self, input):
        embedded = self.embeddings['x'](input)
        embedded_dropout = self.embed_dropout(embedded)
        if self.embeddings_proj:
            embedded_dropout = self.embeddings_proj(embedded_dropout)
        return embedded_dropout

    def init_embed(self, embeddings, **kwargs):
        pdrop = float(kwargs.get('dropout', 0.1))
        self.embed_dropout = nn.Dropout(pdrop)
        self.embeddings = EmbeddingsStack(embeddings)
        input_sz = 0
        for k, embedding in embeddings.items():
            input_sz += embedding.get_dsz()

        projsz = kwargs.get('projsz')
        if projsz:
            self.embeddings_proj = pytorch_linear(input_sz, projsz)
            print('Applying a transform from {} to {}'.format(input_sz, projsz))
            return projsz
        else:
            self.embeddings_proj = None
        return input_sz

    def forward(self, x):
        # the following line masks out the attention to padding tokens
        input_mask = torch.zeros(x.shape, device=x.device, dtype=torch.long).masked_fill(x != 0, 1).unsqueeze(1).unsqueeze(1)
        # the following line builds mask depending on whether it is a causal lm or masked lm
        embedding = self.embed(x)
        embedding = self.proj_to_dsz(embedding)
        transformer_out = self.transformer((embedding, input_mask))
        z = self.get_output(x, transformer_out)
        return z

    def get_output(self, inputs, z):
        return z if self.finetune else z.detach()

    def get_vocab(self):
        return self.vocab

    def get_vsz(self):
        return self.vsz

    def get_dsz(self):
        return self.d_model

    @classmethod
    def load(cls, embeddings, **kwargs):
        c = cls('bert-embed-pooled', **kwargs)
        load_tlm_transformers_bin(c, embeddings)
        return c


@register_embeddings(name='bert-embed')
class BERTEmbeddingsModel(PyTorchEmbeddingsModel, BERTEmbeddings):
    """Register embedding model for usage in mead"""
    pass


def _identity(x):
    return x


def _mean_pool(_, embeddings):
    return torch.mean(embeddings, 1, False)


def _max_pool(_, embeddings):
    return torch.max(embeddings, 1, False)[0]


@register_embeddings(name='bert-embed-pooled')
class BERTPooledEmbeddingsModel(BERTEmbeddingsModel):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

        pooling = kwargs.get('pooling', 'cls')
        if pooling == 'max':
            self.pooling_op = _max_pool
        elif pooling == 'mean':
            self.pooling_op = _mean_pool
        elif pooling == 'sqrt_length':
            self.pooling_op = self._sqrt_length_pool
        else:
            self.pooling_op = self._cls_pool

    def _sqrt_length_pool(self, inputs, embeddings):
        lengths = (inputs != 0).sum(1)
        sqrt_length = lengths.float().sqrt().unsqueeze(1)
        embeddings = embeddings.sum(1) / sqrt_length
        return embeddings

    def _cls_pool(self, inputs, tensor):
        pooled = tensor[inputs == self.cls_index]
        return pooled

    def get_output(self, inputs, z):
        z = self.pooling_op(inputs, z)
        return z if self.finetune else z.detach()

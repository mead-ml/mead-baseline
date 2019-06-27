from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import unicodedata
import six
import numpy as np

from baseline.utils import write_json
from baseline.embeddings import register_embeddings
from baseline.pytorch.embeddings import PyTorchEmbeddings
from baseline.vectorizers import register_vectorizer, AbstractVectorizer
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertModel
from baseline.pytorch.torchy import *
import copy
import json
import math
import re


BERT_TOKENIZER = None


@register_vectorizer(name='wordpiece1d')
class WordPieceVectorizer1D(AbstractVectorizer):

    def __init__(self, **kwargs):
        super(WordPieceVectorizer1D, self).__init__(kwargs.get('transform_fn'))
        global BERT_TOKENIZER
        self.max_seen = 128
        handle = kwargs.get('embed_file')
        if BERT_TOKENIZER is None:
            BERT_TOKENIZER = BertTokenizer.from_pretrained(handle)
        self.tokenizer = BERT_TOKENIZER
        self.mxlen = kwargs.get('mxlen', -1)

    def iterable(self, tokens):
        yield '[CLS]'
        for tok in tokens:
            if tok == '<unk>':
                yield '[UNK]'
            elif tok == '<EOS>':
                yield '[SEP]'
            else:
                for subtok in self.tokenizer.tokenize(tok):
                    yield subtok
        yield '[SEP]'

    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            value = vocab.get(atom)
            if value is None:
                value = vocab['[UNK]']
            yield value

    def run(self, tokens, vocab):
        if self.mxlen < 0:
            self.mxlen = self.max_seen
        vec1d = np.zeros(self.mxlen, dtype=np.long)
        for i, atom in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= 1
                break
            vec1d[i] = atom
        valid_length = i + 1
        return vec1d, valid_length

    def get_dims(self):
        return self.mxlen,


class BERTBaseEmbeddings(PyTorchEmbeddings):

    def __init__(self, name, **kwargs):
        super(BERTBaseEmbeddings, self).__init__(name=name, **kwargs)
        global BERT_TOKENIZER
        self.dsz = kwargs.get('dsz')
        if BERT_TOKENIZER is None:
            BERT_TOKENIZER = BertTokenizer.from_pretrained(kwargs.get('embed_file'))
        self.model = BertModel.from_pretrained(kwargs.get('embed_file'))
        self.vocab = BERT_TOKENIZER.vocab
        self.vsz = len(BERT_TOKENIZER.vocab)  # 30522 self.model.embeddings.word_embeddings.num_embeddings

    def get_vocab(self):
        return self.vocab

    def get_dsz(self):
        return self.dsz

    @classmethod
    def load(cls, embeddings, **kwargs):
        c = cls("bert", **kwargs)
        c.checkpoint = embeddings
        return c

    def forward(self, x):

        input_mask = torch.zeros(x.shape, device=x.device, dtype=torch.long).masked_fill(x != 0, 1)
        input_type_ids = torch.zeros(x.shape, device=x.device, dtype=torch.long)
        all_layers, pooled = self.model(x, token_type_ids=input_type_ids, attention_mask=input_mask)
        z = self.get_output(all_layers, pooled)
        return z

    def get_output(self, all_layers, pooled):
        pass


@register_embeddings(name='bert')
class BERTEmbeddings(BERTBaseEmbeddings):

    def __init__(self, name, **kwargs):
        super(BERTEmbeddings, self).__init__(name=name, **kwargs)
        self.layer_indices = kwargs.get('layers', [-1, -2, -3, -4])
        self.operator = kwargs.get('operator', 'concat')

    def get_output(self, all_layers, pooled):
        layers = [all_layers[layer_index].detach() for layer_index in self.layer_indices]
        z = torch.cat(layers, dim=-1)
        if self.operator != 'concat':
            z = torch.mean(z, dim=-1, keepdim=True)
        return z


@register_embeddings(name='bert-pooled')
class BERTPooledEmbeddings(BERTBaseEmbeddings):

    def __init__(self, name, **kwargs):
        super(BERTPooledEmbeddings, self).__init__(name=name, **kwargs)

    def get_output(self, all_layers, pooled):
        return pooled

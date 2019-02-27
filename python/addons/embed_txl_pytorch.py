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
from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLModel
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from baseline.pytorch.torchy import *
import copy
import json
import math
import re

TXL_TOKENIZER = None


@register_vectorizer(name='txl1d')
class TXLTokenizer1D(AbstractVectorizer):

    def __init__(self, **kwargs):
        super(TXLTokenizer1D, self).__init__(kwargs.get('transform_fn'))
        global TXL_TOKENIZER
        self.max_seen = 128
        handle = kwargs.get('embed_file')
        if TXL_TOKENIZER is None:
            TXL_TOKENIZER = TransfoXLTokenizer.from_pretrained(handle)
        self.tokenizer = TXL_TOKENIZER
        self.mxlen = kwargs.get('mxlen', -1)

    def iterable(self, tokens):
        for tok in tokens:
            for subtok in self.tokenizer.tokenize(tok):
                yield subtok

    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            value = vocab.get(atom)
            if value is None:
                value = vocab['<unk>']
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


@register_embeddings(name='txl')
class TXLEmbeddings(PyTorchEmbeddings):

    def __init__(self, name, **kwargs):
        super(TXLEmbeddings, self).__init__(name=name, **kwargs)
        global TXL_TOKENIZER
        self.dsz = kwargs.get('dsz')
        if TXL_TOKENIZER is None:
            TXL_TOKENIZER = TransfoXLTokenizer.from_pretrained(kwargs.get('embed_file'))
        self.model = TransfoXLModel.from_pretrained(kwargs.get('embed_file'))
        self.vocab = TXL_TOKENIZER.sym2idx
        self.vsz = len(TXL_TOKENIZER.sym2idx)

    def get_vocab(self):
        return self.vocab

    def get_dsz(self):
        return self.dsz

    @classmethod
    def load(cls, embeddings, **kwargs):
        c = cls("txl", **kwargs)
        c.checkpoint = embeddings
        return c

    def forward(self, x):
        with torch.no_grad():
            last_hidden_layer, _ = self.model(x)
            last_hidden_layer = last_hidden_layer.detach()
        return last_hidden_layer

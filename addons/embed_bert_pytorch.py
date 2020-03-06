from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import unicodedata
import six
import numpy as np
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
        super().__init__(kwargs.get('transform_fn'))
        global BERT_TOKENIZER
        self.max_seen = 128
        handle = kwargs.get('embed_file')
        if 'uncased' in handle or bool(kwargs.get('do_lower_case', False)) is False:
            do_lower_case = False
        else:
            do_lower_case = True
        if BERT_TOKENIZER is None:
            BERT_TOKENIZER = BertTokenizer.from_pretrained(handle, do_lower_case=do_lower_case)
        self.tokenizer = BERT_TOKENIZER
        self.mxlen = kwargs.get('mxlen', -1)

    def count(self, tokens):
        seen = 0
        counter = collections.Counter()
        for tok in self.iterable(tokens):
            counter[tok] += 1
            seen += 1
        self.max_seen = max(self.max_seen, seen)
        return counter

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


@register_vectorizer(name='wordpiece-label-dict1d')
class WordPieceLabelDict1DVectorizer(WordPieceVectorizer1D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field = kwargs.get('fields', kwargs.get('field', 'text'))
        self.label = kwargs.get('label', 'label')

    def iterable(self, tokens):
        yield Offsets.VALUES[Offsets.PAD]
        for t in tokens:
            t_word = t[self.field]
            t_label = t[self.label]
            subwords = [x for x in self.tokenizer.tokenize(t_word)]
            subwords = [Offsets.VALUES[Offsets.PAD]] * len(subwords)
            # TODO: The tokenizer sometimes cuts up the token and leaves nothing
            # how to handle this since we cannot get anything for it
            if len(subwords):
                subwords[0] = t_label
            for x in subwords:
                yield x
        yield Offsets.VALUES[Offsets.PAD]

    def run(self, tokens, vocab):
        return super().run(tokens, vocab)

    def count(self, tokens):
        seen = 0
        counter = collections.Counter()
        for tok in self.iterable(tokens):
            counter[tok] += 1
            seen += 1
        self.max_seen = max(self.max_seen, seen)
        return counter


@register_vectorizer(name='wordpiece-dict1d')
class WordPieceDict1DVectorizer(WordPieceVectorizer1D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field = kwargs.get('fields', kwargs.get('field', 'text'))
        self.delim = kwargs.get('token_delim', '~~')

    def iterable(self, tokens):
        yield '[CLS]'
        for t in tokens:
            tok = t[self.field]
            for subtok in self.tokenizer.tokenize(tok):
                yield subtok
        yield '[SEP]'


class BERTBaseEmbeddings(PyTorchEmbeddings):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        global BERT_TOKENIZER
        self.dsz = kwargs.get('dsz')
        handle = kwargs.get('embed_file')
        if BERT_TOKENIZER is None:
            if 'uncased' in handle or bool(kwargs.get('do_lower_case', False)) is False:
                do_lower_case = False
            else:
                do_lower_case = True
            BERT_TOKENIZER = BertTokenizer.from_pretrained(handle, do_lower_case=do_lower_case)
        self.model = BertModel.from_pretrained(kwargs.get('embed_file'))
        self.vocab = BERT_TOKENIZER.vocab
        self.vsz = len(BERT_TOKENIZER.vocab)  # 30522 self.model.embeddings.word_embeddings.num_embeddings

    def get_vocab(self):
        return self.vocab

    def get_dsz(self):
        return self.dsz

    def get_vsz(self):
        return self.vsz

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
        """BERT sequence embeddings, used for a feature-ful representation of finetuning sequence tasks.

        If operator == 'concat' result is [B, T, #Layers * H] other size the layers are mean'd the shape is [B, T, H]
        """
        super().__init__(name=name, **kwargs)
        self.layer_indices = kwargs.get('layers', [-1, -2, -3, -4])
        self.operator = kwargs.get('operator', 'concat')
        self.finetune = kwargs.get('finetune', False)

    def get_output(self, all_layers, pooled):
        if self.finetune:
            layers = [all_layers[layer_index] for layer_index in self.layer_indices]
        else:
            layers = [all_layers[layer_index].detach() for layer_index in self.layer_indices]
        if self.operator != 'concat':
            z = torch.cat([l.unsqueeze(-1) for l in layers], dim=-1)
            z = torch.mean(z, dim=-1)
        else:
            z = torch.cat(layers, dim=-1)
        return z

    def extra_repr(self):
        return f"finetune={self.finetune}, combination={self.operator}, layers={self.layer_indices}"


@register_embeddings(name='bert-pooled')
class BERTPooledEmbeddings(BERTBaseEmbeddings):

    def __init__(self, name, **kwargs):
        super(BERTPooledEmbeddings, self).__init__(name=name, **kwargs)

    def get_output(self, all_layers, pooled):
        return pooled


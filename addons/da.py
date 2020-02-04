from baseline.reader import SeqPredictReader, register_reader
from baseline.utils import Offsets, listify
#from eight_mile.pytorch.embeddings import register_embeddings, LookupTableEmbeddings
from glob import iglob
import os
import pandas as pd
#import torch
from baseline.vectorizers import register_vectorizer, AbstractVectorizer, _token_iterator
import collections
from nltk import word_tokenize
import numpy as np


@register_vectorizer(name='turn2d')
class Turn2DVectorizer(AbstractVectorizer):

    def __init__(self, **kwargs):
        super().__init__(kwargs.get('transform_fn', word_tokenize))
        self.mxlen = kwargs.get('mxlen', -1)
        self.mxwlen = kwargs.get('mxwlen', -1)
        self.max_seen_turns = 0
        self.max_seen_words = 0

    def _next_element(self, turns, vocab):

        OOV = vocab['<UNK>']
        EOT = vocab.get('<EOT>', vocab.get(' ', Offsets.PAD))

        for turn in turns:
            for word in word_tokenize(turn['text']):
                yield vocab.get(word, OOV)
            yield EOT

    def count(self, turns):
        seen_turn = 0
        counter = collections.Counter()
        for turn in self.iterable(turns):
            self.max_seen_words = max(self.max_seen_words, len(turn))
            seen_turn += 1
            for word in turn:
                counter[word] += 1
            counter['<EOT>'] += 1
        self.max_seen_turns = max(self.max_seen_turns, seen_turn)
        return counter

    def reset(self):
        self.mxlen = -1
        self.mxwlen = -1
        self.max_seen_turns = 0
        self.max_seen_words = 0

    def run(self, tokens, vocab):

        if self.mxlen < 0:
            self.mxlen = self.max_seen_turns
        if self.mxwlen < 0:
            self.mxwlen = self.max_seen_words

        EOT = vocab.get('<EOT>', vocab.get(' ', Offsets.PAD))

        vec2d = np.zeros((self.mxlen, self.mxwlen), dtype=int)
        i = 0
        j = 0
        over = False
        for atom in self._next_element(tokens, vocab):
            if over:
                # If if we have gone over mxwlen burn tokens until we hit end of word
                if atom == EOT:
                    over = False
                continue
            if i == self.mxlen:
                break
            if atom == EOT:
                i += 1
                j = 0
                continue
            elif j == self.mxwlen:
                over = True
                i += 1
                j = 0
                continue
            else:
                vec2d[i, j] = atom
                j += 1
        valid_length = i
        return vec2d, valid_length

    def get_dims(self):
        return self.mxlen, self.mxwlen


@register_vectorizer(name='turn-dict2d')
class DictTurn2DVectorizer(Turn2DVectorizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fields = listify(kwargs.get('fields', 'text'))
        self.delim = kwargs.get('token_delim', '@@')

    def iterable(self, tokens):

        for tok in tokens:
            token = []
            for field in self.fields:
                if isinstance(tok, dict):
                    token += self.transform_fn(tok[field])
                else:
                    token += self.transform_fn(tok)
            yield token


@register_reader(task='tagger', name='da')
class DASeqReader(SeqPredictReader):

    def __init__(self, vectorizers, trim=False, truncate=False, mxlen=-1, utt_idx=1, da_idx=2, **kwargs):
        super().__init__(vectorizers, trim, truncate, mxlen, **kwargs)
        self.utt_idx = utt_idx
        self.da_idx = da_idx

    def read_examples(self, tsfile):

        dlgs = []
        for f in iglob(f'{tsfile}/*.txt'):
            print(f)
            if not os.path.isfile(f):
                continue

            dlg = pd.read_csv(f, sep='|', header=None, dtype=str, na_filter=False)

            da_idx = dlg.columns[self.da_idx] if self.da_idx < 0 else self.da_idx
            utt_idx = dlg.columns[self.utt_idx] if self.utt_idx < 0 else self.utt_idx

            dlg_utt = dlg[utt_idx]
            dlg_da = dlg[da_idx]
            turns = []
            for utt, da in zip(dlg_utt, dlg_da):
                turn = {'text': utt, 'y': da}
                turns.append(turn)
            dlgs.append(turns)
        return dlgs


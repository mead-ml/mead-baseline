import numpy as np
from collections import Counter
from baseline.utils import Offsets
from embed_bert import WordpieceTokenizer, WordPieceVectorizer1D
from baseline.vectorizers import register_vectorizer, AbstractVectorizer


@register_vectorizer(name='wordpiece1dlm')
class WordPieceVectorizer1DLM(WordPieceVectorizer1D):
    # override iterable function for continuous sequence data (typically for lm)
    def iterable(self, tokens):
        for tok in tokens:
            if tok == '<unk>':
                yield '[UNK]'
            elif tok == '<EOS>':
                yield '[SEP]'
            else:
                for subtok in self.wordpiece_tok.tokenize(tok):
                    yield subtok


@register_vectorizer(name='bpe1d')
class BPEVectorizer1D(AbstractVectorizer):
    """Define a Baseline Vectorizer for BPE using fastBPE (https://github.com/glample/fastBPE)

    If you use tokens=bpe, this vectorizer is used, and so then there is a
    dependency on fastBPE

    To use BPE, we assume that a Dictionary of codes and vocab was already created

    """
    def __init__(self, **kwargs):
        """Loads a BPE tokenizer"""
        super(BPEVectorizer1D, self).__init__(kwargs.get('transform_fn'))
        from fastBPE import fastBPE
        self.max_seen = 128
        self.model_file = kwargs.get('model_file')
        self.vocab_file = kwargs.get('vocab_file')
        self.tokenizer = fastBPE(self.model_file, self.vocab_file)
        self.mxlen = kwargs.get('mxlen', -1)
        self.vocab = {k: i for i, k in enumerate(self.read_vocab(self.vocab_file))}

    def read_vocab(self, s):
        vocab = [] + Offsets.VALUES
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
            if t == '<unk>':
                yield Offsets.UNK
            if t == '<eos>':
                yield Offsets.EOS
            else:
                subwords = self.tokenizer.apply([t])[0].split()
                for x in subwords:
                    yield x

    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            value = vocab.get(atom, 0)  # This shouldnt actually happen
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

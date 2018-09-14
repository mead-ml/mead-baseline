import numpy as np
from baseline.utils import export, create_user_vectorizer
from baseline.data import reverse_2nd
import collections


__all__ = []
exporter = export(__all__)


class Vectorizer(object):

    def __init__(self):
        pass

    def _iterable(self, tokens):
        for tok in tokens:
            yield tok

    def _next_element(self, tokens, vocab):
        OOV = vocab['<UNK>']
        for atom in self._iterable(tokens):
            yield vocab.get(atom, OOV)

    def count(self, tokens):
        counter = collections.Counter()
        for tok in self._iterable(tokens):
            counter[tok] += 1
        return counter

    def run(self, tokens, vocab):
        pass


class Token1DVectorizer(Vectorizer):

    def __init__(self, **kwargs):
        super(Vectorizer, self).__init__()
        self.mxlen = kwargs.get('mxlen', kwargs.get('maxs', 100))
        self.time_reverse = kwargs.get('rev', False)

    def run(self, tokens, vocab):
        vec1d = np.zeros(self.mxlen, dtype=int)
        for i, atom in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= 1
                break
            vec1d[i] = atom
        valid_length = i

        if self.time_reverse:
            vec1d = reverse_2nd(vec1d)
        return vec1d, valid_length


class AbstractCharVectorizer(Vectorizer):

    def __init__(self):
        super(AbstractCharVectorizer, self).__init__()

    def _next_element(self, tokens, vocab):
        OOV = vocab['<UNK>']
        EOW = vocab.get('<EOW>', vocab.get(' '))

        for token in self._iterable(tokens):
            for ch in token:
                yield self.vocab.get(ch, OOV)
            yield EOW


class Char2DLookupVectorizer(AbstractCharVectorizer):

    def __init__(self, **kwargs):
        super(Char2DLookupVectorizer, self).__init__()
        self.mxlen = kwargs.get('mxlen', kwargs.get('maxs', 100))
        self.mxwlen = kwargs.get('mxwlen', kwargs.get('maxw', 40))

    def run(self, tokens, vocab):
        vec2d = np.zeros((self.mxlen, self.mxwlen), dtype=int)
        i = 0
        j = 0
        for atom in self._next_element(tokens):
            if i == self.mxlen:
                i -= 1
                break
            if atom == self.EOW or j == self.mxwlen:
                i += 1
                j = 0
            else:
                vec2d[i, j] = atom
        valid_length = i
        return vec2d, valid_length


class Char1DLookupVectorizer(AbstractCharVectorizer):

    def __init__(self, **kwargs):
        super(Char1DLookupVectorizer, self).__init__()
        self.mxlen = kwargs.get('mxlen', kwargs.get('maxs', 100))
        self.time_reverse = kwargs.get('rev', False)

    def run(self, tokens, vocab):

        vec1d = np.zeros(self.mxlen, dtype=int)
        for i, atom in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= 1
            vec1d[i] = atom
        valid_length = i
        if self.time_reverse:
            vec1d = reverse_2nd(vec1d)
        return vec1d, valid_length


BASELINE_KNOWN_VECTORIZERS = {
    'token1d': Token1DVectorizer,
    'char2d': Char2DLookupVectorizer,
    'char1d': Char1DLookupVectorizer
}


@exporter
def create_vectorizer(filename, known_vocab=None, **kwargs):
    vec_type = kwargs.get('vectorizer_type', kwargs.get('type', 'token1d'))
    Constructor = BASELINE_KNOWN_VECTORIZERS.get(vec_type)
    if Constructor is not None:
        return Constructor(**kwargs)
    else:
        print('loading user module')
    return create_user_vectorizer(filename, known_vocab, **kwargs)

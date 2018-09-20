import numpy as np
from baseline.utils import export, create_user_vectorizer, listify
import collections


__all__ = []
exporter = export(__all__)


@exporter
class Vectorizer(object):

    def __init__(self):
        pass

    def run(self, tokens, vocab):
        pass

    def count(self, tokens):
        pass

    def get_dims(self):
        pass


@exporter
def identity_trans_fn(x):
    return x


@exporter
class AbstractVectorizer(Vectorizer):

    def __init__(self, transform_fn=None):
        super(AbstractVectorizer, self).__init__()
        self.transform_fn = identity_trans_fn if transform_fn is None else transform_fn
        #print(self.transform_fn)

    def _iterable(self, tokens):
        for tok in self.transform_fn(tokens):
            yield tok

    def _next_element(self, tokens, vocab):
        for atom in self._iterable(tokens):
            value = vocab.get(atom)
            if value is None:
                value = vocab['<UNK>']
            yield value


@exporter
class Token1DVectorizer(AbstractVectorizer):

    def __init__(self, **kwargs):
        super(Token1DVectorizer, self).__init__(kwargs.get('transform_fn'))
        self.time_reverse = kwargs.get('rev', False)
        self.mxlen = kwargs.get('mxlen', -1)
        self.max_seen = 0

    def count(self, tokens):
        seen = 0
        counter = collections.Counter()
        for tok in self._iterable(tokens):
            counter[tok] += 1
            seen += 1
            counter['<EOW>'] += 1
        self.max_seen = max(self.max_seen, seen)
        return counter

    def run(self, tokens, vocab):

        if self.mxlen < 0:
            self.mxlen = self.max_seen

        vec1d = np.zeros(self.mxlen, dtype=int)
        for i, atom in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= 1
                break
            vec1d[i] = atom
        valid_length = i + 1

        if self.time_reverse:
            vec1d = vec1d[::-1]
            return vec1d, None
        return vec1d, valid_length

    def get_dims(self):
        return self.mxlen,


@exporter
class GOVectorizer(Vectorizer):

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        if self.vectorizer.mxlen != -1:
            self.vectorizer.mxlen -= 1

    def count(self, tokens):
        counter = self.vectorizer.count(tokens)
        counter['<GO>'] += 1
        return counter

    def run(self, tokens, vocab):
        GO = vocab['<GO>']
        EOS = vocab['<EOS>']
        vec1d, valid_length = self.vectorizer.run(tokens, vocab)
        vec1d = np.concatenate([[GO], vec1d])
        vec1d[valid_length] = EOS
        return vec1d, valid_length + 1

    def get_dims(self):
        return self.vectorizer.get_dims()[0] + 1,


def _token_iterator(vectorizer, tokens):
    for tok in tokens:
        token = []
        for field in vectorizer.fields:
            token += [vectorizer.transform_fn(tok[field])]
        yield vectorizer.delim.join(token)


@exporter
class Dict1DVectorizer(Token1DVectorizer):

    def __init__(self, **kwargs):
        super(Dict1DVectorizer, self).__init__(**kwargs)
        self.fields = listify(kwargs.get('fields', 'text'))
        self.delim = kwargs.get('token_delim', '@@')

    def _iterable(self, tokens):
        return _token_iterator(self, tokens)


@exporter
class AbstractCharVectorizer(AbstractVectorizer):

    def __init__(self, transform_fn=None):
        super(AbstractCharVectorizer, self).__init__(transform_fn)

    def _next_element(self, tokens, vocab):
        OOV = vocab['<UNK>']
        EOW = vocab.get('<EOW>', vocab.get(' '))
        for token in self._iterable(tokens):
            for ch in token:
                yield vocab.get(ch, OOV)
            yield EOW


@exporter
class Char2DVectorizer(AbstractCharVectorizer):

    def __init__(self, **kwargs):
        super(Char2DVectorizer, self).__init__(kwargs.get('transform_fn'))
        self.mxlen = kwargs.get('mxlen', -1)
        self.mxwlen = kwargs.get('mxwlen', -1)
        self.max_seen_tok = 0
        self.max_seen_char = 0

    def count(self, tokens):
        seen_tok = 0
        counter = collections.Counter()
        for token in self._iterable(tokens):
            self.max_seen_char = max(self.max_seen_char, len(token))
            seen_tok += 1
            for ch in token:
                counter[ch] += 1
            counter['<EOW>'] += 1
        self.max_seen_tok = max(self.max_seen_tok, seen_tok)
        return counter

    def run(self, tokens, vocab):

        if self.mxlen < 0:
            self.mxlen = self.max_seen_tok
        if self.mxwlen < 0:
            self.mxwlen = self.max_seen_char

        EOW = vocab.get('<EOW>', vocab.get(' '))

        vec2d = np.zeros((self.mxlen, self.mxwlen), dtype=int)
        i = 0
        j = 0
        for atom in self._next_element(tokens, vocab):
            if i == self.mxlen:
                i -= 1
                break
            if atom == EOW or j == self.mxwlen:
                i += 1
                j = 0
            else:
                vec2d[i, j] = atom
                j += 1
        valid_length = i + 1
        return vec2d, valid_length

    def get_dims(self):
        return self.mxlen, self.mxwlen


@exporter
class Dict2DVectorizer(Char2DVectorizer):

    def __init__(self, **kwargs):
        super(Dict2DVectorizer, self).__init__(**kwargs)
        self.fields = listify(kwargs.get('fields', 'text'))
        self.delim = kwargs.get('token_delim', '@@')

    def _iterable(self, tokens):
        return _token_iterator(self, tokens)


@exporter
class Char1DVectorizer(AbstractCharVectorizer):

    def __init__(self, **kwargs):
        super(Char1DVectorizer, self).__init__(kwargs.get('transform_fn'))
        print(kwargs)
        self.mxlen = kwargs.get('mxlen', -1)
        self.time_reverse = kwargs.get('rev', False)
        self.max_seen_tok = 0

    def count(self, tokens):
        seen_tok = 0
        counter = collections.Counter()
        for token in self._iterable(tokens):
            seen_tok += 1
            for ch in token:
                counter[ch] += 1
                seen_tok += 1
            counter['<EOW>'] += 1
            seen_tok += 1

        self.max_seen_tok = max(self.max_seen_tok, seen_tok)
        return counter

    def run(self, tokens, vocab):

        if self.mxlen < 0:
            self.mxlen = self.max_seen_char

        vec1d = np.zeros(self.mxlen, dtype=int)
        for i, atom in enumerate(self._next_element(tokens, vocab)):
            if i == self.mxlen:
                i -= 1
            vec1d[i] = atom
        if self.time_reverse:
            vec1d = vec1d[::-1]
            return vec1d, None
        return vec1d, i + 1

    def get_dims(self):
        return self.mxlen,


BASELINE_KNOWN_VECTORIZERS = {
    'token1d': Token1DVectorizer,
    'char2d': Char2DVectorizer,
    'char1d': Char1DVectorizer,
    'dict1d': Dict1DVectorizer,
    'dict2d': Dict2DVectorizer
}


@exporter
def create_vectorizer(**kwargs):
    vec_type = kwargs.get('vectorizer_type', kwargs.get('type', 'token1d'))
    print(vec_type)
    Constructor = BASELINE_KNOWN_VECTORIZERS.get(vec_type)
    if Constructor is not None:
        return Constructor(**kwargs)
    else:
        print('loading user module')
    return create_user_vectorizer(**kwargs)

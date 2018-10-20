import numpy as np
from baseline.vectorizers import Token1DVectorizer, register_vectorizer


@register_vectorizer(name='text')
class Text1DVectorizer(Token1DVectorizer):
    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            yield atom

    def run(self, tokens, vocab):
        if self.mxlen < 0:
            self.mxlen = self.max_seen
        vec1d = np.full(self.mxlen, '', dtype=np.object)
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

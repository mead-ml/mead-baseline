from baseline.featurizers import Featurizer
from baseline.utils import lowercase


class TaggerElmoFeaturizer(Featurizer):
    def __init__(self, tagger, mxlen, maxw, zero_alloc, **kwargs):
        super(TaggerElmoFeaturizer, self).__init__(tagger, mxlen, maxw, zero_alloc)

    def run(self, tokens):
        tokens = [token[0] for token in tokens]
        xs = self.zero_alloc((1, self.mxlen), dtype=int)
        xs_lc = self.zero_alloc((1, self.mxlen), dtype=int)
        xs_ch = self.zero_alloc((1, self.mxlen, self.maxw), dtype=int)
        lengths = self.zero_alloc(1, dtype=int)
        lengths[0] = min(len(tokens), self.mxlen)
        words_vocab = self.model.get_vocab(vocab_type='word')
        chars_vocab = self.model.get_vocab(vocab_type='char')
        for j in range(self.mxlen):
            if j == len(tokens):
                break
            w = tokens[j]
            nch = min(len(w), self.maxw)
            xs[0, j] = words_vocab.get(w, 0)
            xs_lc[0, j] = words_vocab.get(lowercase(w), 0)
            for k in range(nch):
                xs_ch[0, j, k] = chars_vocab.get(w[k], 0)
        return {'x': xs, 'x_lc': xs_lc, 'xch': xs_ch, 'lengths': lengths}


def create_featurizer(model, mxlen, maxw, zero_alloc, **kwargs):
    return TaggerElmoFeaturizer(model, mxlen, maxw, zero_alloc)

import numpy as np
from baseline.utils import lowercase, import_user_module


class Featurizer(object):
    def __init__(self, model, mxlen, maxw, zero_alloc, word_trans_fn, **kwargs):
        self.mxlen = mxlen
        self.maxw = maxw
        self.zero_alloc = zero_alloc
        self.word_trans_fn = word_trans_fn
        self.model = model

    def featurize(self, tokens):
        pass


class TaggerDefaultFeaturizer(Featurizer):
    def __init__(self, tagger, mxlen, maxw, zero_alloc, word_trans_fn):
        super(TaggerDefaultFeaturizer, self).__init__(tagger, mxlen, maxw, zero_alloc, word_trans_fn)

    def featurize(self, tokens):
        tokens = [token[0] for token in tokens]
        xs = self.zero_alloc((1, self.mxlen), dtype=int)
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
            xs[0, j] = words_vocab.get(self.word_trans_fn(w), 0)
            for k in range(nch):
                xs_ch[0, j, k] = chars_vocab.get(w[k], 0)
        return {'x': xs, 'xch': xs_ch, 'lengths': lengths}


class TaggerMultiFeatureFeaturizer(Featurizer):
    def __init__(self, tagger, mxlen, maxw, zero_alloc, word_trans_fn, vocab_keys):
        super(TaggerMultiFeatureFeaturizer, self).__init__(tagger, mxlen, maxw, zero_alloc, word_trans_fn)
        self.vocab_keys = vocab_keys

    def featurize(self, tokens):
        xs = self.zero_alloc((1, self.mxlen), dtype=int)
        xs_ch = self.zero_alloc((1, self.mxlen, self.maxw), dtype=int)
        lengths = self.zero_alloc(1, dtype=int)
        lengths[0] = min(len(tokens), self.mxlen)
        data = {}
        for j in range(self.mxlen):
            if j == len(tokens):
                break
            token_features = tokens[j]
            if 'word' in self.vocab_keys:
                word_index = self.vocab_keys['word']
                words_vocab = self.model.get_vocab(vocab_type='word')
                w = token_features[word_index]
                xs[0, j] = words_vocab.get(self.word_trans_fn(w), 0)
                if 'char' in self.vocab_keys:
                    nch = min(len(w), self.maxw)
                    for k in range(nch):
                        chars_vocab = self.model.get_vocab(vocab_type='char')
                        xs_ch[0, j, k] = chars_vocab.get(w[k], 0)
            for key in self.vocab_keys:
                if not key == 'word' and not key == 'char':
                    feature_index = self.vocab_keys[key]
                    feature = token_features[feature_index]
                    feature_vocab = self.model.get_vocab(vocab_type=key)
                    data[key] = self.zero_alloc((1, self.mxlen), dtype=np.int)
                    data[key][0, j] = feature_vocab[feature]
        data.update({'x': xs, 'xch': xs_ch, 'lengths': lengths})
        return data


def create_featurizer(model, mxlen, maxw, zero_alloc=np.zeros, word_trans_fn=lowercase, **kwargs):
    featurizer_type = kwargs.get('featurizer_type', 'default')
    if featurizer_type == 'default':
        return TaggerDefaultFeaturizer(model, mxlen, maxw, zero_alloc, word_trans_fn)
    elif featurizer_type == 'multifeature':
        return TaggerMultiFeatureFeaturizer(model, mxlen, maxw, zero_alloc, word_trans_fn, kwargs['vocab_keys'])
    else:
        mod = import_user_module("featurizer", featurizer_type)
        return mod.create_featurizer(model, mxlen, maxw, zero_alloc, word_trans_fn, **kwargs)

import numpy as np
from baseline.utils import import_user_module, lowercase


class Featurizer(object):
    def __init__(self, model, mxlen, maxw, zero_alloc):
        self.mxlen = mxlen
        self.maxw = maxw
        self.zero_alloc = zero_alloc
        self.model = model

    def run(self, tokens):
        pass


class WordCharLength(Featurizer):
    def __init__(self, tagger, mxlen, maxw, zero_alloc, **kwargs):
        super(WordCharLength, self).__init__(tagger, mxlen, maxw, zero_alloc)
        self.word_trans_fn = kwargs.get('word_trans_fn', lowercase)

    def run(self, tokens):
        if type(tokens[0]) is not str:
            tokens = [token[0] for token in tokens]
        xs = self.zero_alloc((1, self.mxlen), dtype=int)

        chars_vocab = self.model.get_vocab('char')
        if chars_vocab is not None:
            if self.maxw is None:
                raise Exception('Expected max word length')
            xs_ch = self.zero_alloc((1, self.mxlen, self.maxw), dtype=int)
        else:
            xs_ch = None
        lengths = self.zero_alloc(1, dtype=int)
        lengths[0] = min(len(tokens), self.mxlen)
        words_vocab = self.model.get_vocab('word')
        for j in range(self.mxlen):
            if j == len(tokens):
                break
            w = tokens[j]
            xs[0, j] = words_vocab.get(self.word_trans_fn(w), 0)

            if chars_vocab is not None:
                nch = min(len(w), self.maxw)

                for k in range(nch):
                    xs_ch[0, j, k] = chars_vocab.get(w[k], 0)
        return {'x': xs, 'xch': xs_ch, 'lengths': lengths}


class MultiFeatureFeaturizer(Featurizer):
    def __init__(self, tagger, mxlen, maxw, zero_alloc, **kwargs):
        super(MultiFeatureFeaturizer, self).__init__(tagger, mxlen, maxw, zero_alloc)
        self.vocab_keys = kwargs['vocab_keys']
        self.word_trans_fn = kwargs.get('word_trans_fn', lowercase)

    def run(self, tokens):
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
                words_vocab = self.model.get_vocab('word')
                w = token_features[word_index]
                xs[0, j] = words_vocab.get(self.word_trans_fn(w), 0)
                if 'char' in self.vocab_keys:
                    nch = min(len(w), self.maxw)
                    for k in range(nch):
                        chars_vocab = self.model.get_vocab('char')
                        xs_ch[0, j, k] = chars_vocab.get(w[k], 0)
            for key in self.vocab_keys:
                if not key == 'word' and not key == 'char':
                    feature_index = self.vocab_keys[key]
                    feature = token_features[feature_index]
                    feature_vocab = self.model.get_vocab(key)
                    data[key] = self.zero_alloc((1, self.mxlen), dtype=np.int)
                    data[key][0, j] = feature_vocab[feature]
        data.update({'x': xs, 'xch': xs_ch, 'lengths': lengths})
        return data


def create_featurizer(model, zero_alloc=np.zeros, **kwargs):

    mxlen = kwargs.pop('mxlen', model.mxlen if hasattr(model, 'mxlen') else -1)
    maxw = kwargs.pop('maxw', model.maxw if hasattr(model, 'maxw') else model.mxwlen if hasattr(model, 'mxwlen') else -1)
    kwargs.pop('zero_alloc', None)
    featurizer_type = kwargs.get('featurizer_type', 'default')
    if featurizer_type == 'default':
        return WordCharLength(model, mxlen, maxw, zero_alloc, **kwargs)
    elif featurizer_type == 'multifeature':
        return MultiFeatureFeaturizer(model, mxlen, maxw, zero_alloc, **kwargs)
    else:
        mod = import_user_module("featurizer", featurizer_type)
        return mod.create_featurizer(model, mxlen, maxw, zero_alloc, **kwargs)

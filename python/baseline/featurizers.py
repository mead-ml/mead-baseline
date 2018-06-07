def tagger_featurizer(tagger, tokens, mxlen, maxw, zero_alloc, word_trans_fn, vocab_keys):
    xs = zero_alloc((1, mxlen), dtype=int)
    xs_ch = zero_alloc((1, mxlen, maxw), dtype=int)
    lengths = zero_alloc(1, dtype=int)
    lengths[0] = min(len(tokens), mxlen)
    data = {}
    if not type(tokens[0]) is list:  # support the existing case
        tokens = [[token] for token in tokens]
    for j in range(mxlen):
        if j == len(tokens):
            break
        token_features = tokens[j]
        if 'word' in vocab_keys:
            word_index = vocab_keys['word']
            words_vocab = tagger.get_vocab(vocab_type='word')
            w = token_features[word_index]
            xs[0, j] = words_vocab.get(word_trans_fn(w), 0)
            if 'char' in vocab_keys:
                nch = min(len(w), maxw)
                for k in range(nch):
                    chars_vocab = tagger.get_vocab(vocab_type='char')
                    xs_ch[0, j, k] = chars_vocab.get(w[k], 0)
        for key in vocab_keys:
            if not key == 'word' and not key == 'char':
                feature_index = vocab_keys[key]
                feature = token_features[feature_index]
                feature_vocab = tagger.get_vocab(vocab_type=key)
                data[key] = zero_alloc((1, mxlen), dtype=np.int)
                data[key][0, j] = feature_vocab[feature]
    data.update({'x': xs, 'xch': xs_ch, 'lengths': lengths})
    return data


def tagger_featurizer_elmo(tagger, tokens, mxlen, maxw, zero_alloc, word_trans_fn, vocab_keys):
    xs = zero_alloc((1, mxlen), dtype=int)
    xs_lc = zero_alloc((1, mxlen), dtype=int)
    xs_ch = zero_alloc((1, mxlen, maxw), dtype=int)
    lengths = zero_alloc(1, dtype=int)
    lengths[0] = min(len(tokens), mxlen)
    data = {}
    if not type(tokens[0]) is list:  # support the existing case
        tokens = [[token] for token in tokens]
    for j in range(mxlen):
        if j == len(tokens):
            break
        token_features = tokens[j]
        if 'word' in vocab_keys:
            word_index = vocab_keys['word']
            words_vocab = tagger.get_vocab(vocab_type='word')
            w = token_features[word_index]
            xs[0, j] = words_vocab.get(w, 0)
            xs_lc[0, j] = words_vocab.get(w.lower(), 0)
            if 'char' in vocab_keys:
                nch = min(len(w), maxw)
                for k in range(nch):
                    chars_vocab = tagger.get_vocab(vocab_type='char')
                    xs_ch[0, j, k] = chars_vocab.get(w[k], 0)
        for key in vocab_keys:
            if not key == 'word' and not key == 'char':
                feature_index = vocab_keys[key]
                feature = token_features[feature_index]
                feature_vocab = tagger.get_vocab(vocab_type=key)
                data[key] = zero_alloc((1, mxlen), dtype=np.int)
                data[key][0, j] = feature_vocab[feature]
    data.update({'x': xs, 'x_lc': xs_lc, 'xch': xs_ch, 'lengths': lengths})
    return data


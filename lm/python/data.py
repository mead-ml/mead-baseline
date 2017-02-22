import w2v
import numpy as np
from collections import Counter
import codecs

def ptb_build_vocab(files):
    vocab_word = Counter()
    vocab_ch = Counter()
    maxw = 0
    num_words_in_files = []
    for file in files:
        if file is None:
            continue

        with codecs.open(file, encoding='utf-8', mode='r') as f:
            num_words = 0
            for line in f:
                sentence = line.split() + ['<EOS>']
                num_words += len(sentence)
                for w in sentence:
                    vocab_word[w] += 1
                    maxw = max(maxw, len(w))
                    for k in w:
                        vocab_ch[k] += 1
            num_words_in_files.append(num_words)

    return maxw, vocab_ch, vocab_word, num_words_in_files

def ptb_load_sentences(filename, words_vocab, chars_vocab, num_words, maxw, vec_alloc=np.zeros):

    xch = vec_alloc((num_words, maxw), np.int)
    x = vec_alloc((num_words), np.int)
    i = 0
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        for line in f:
            sentence = line.split() + ['<EOS>']
            num_words += len(sentence)
            for w in sentence:
                x[i] = words_vocab.get(w)
                nch = min(len(w), maxw)
                for k in range(nch):
                    xch[i,k] = chars_vocab.get(w[k], 0)
                i += 1

    return x, xch

def batch(ts, nbptt, batchsz, wsz):

    x = ts[0]
    xch = ts[1]
    num_examples = x.shape[0]
    rest = num_examples // batchsz
    steps = rest // nbptt
    stride_ch = nbptt * wsz
    trunc = batchsz * rest

    print('Truncating from %d to %d' % (num_examples, trunc))
    x = x[:trunc].reshape((batchsz, rest))
    xch = xch.flatten()
    trunc = batchsz * rest * wsz

    print('Truncated from %d to %d' % (xch.shape[0], trunc))
    xch = xch[:trunc].reshape((batchsz, rest * wsz))

    for i in range(steps):
        yield x[:, i*nbptt:(i+1)*nbptt].reshape((batchsz, nbptt)), \
            xch[:, i*stride_ch:(i+1)*stride_ch].reshape((batchsz, nbptt, wsz)), \
            x[:, i*nbptt+1:(i+1)*nbptt+1].reshape((batchsz, nbptt))


def num_steps_per_epoch(num_examples, nbptt, batchsz):
    rest = num_examples // batchsz
    return rest // nbptt


def show_batch_words(batch, word_lut, limit=1000):
    for x in batch[0]:
        print(' '.join([word_lut[x[i]] for i in range(min(limit, x.shape[0]))]))


def show_batch_letters(batch, char_lut, word_limit=1000, char_limit = 200):
    for xch in batch[1]:
        print('[')
        for i in range(min(word_limit,xch.shape[0])):
            xch_i = xch[i]
            word_repr = ' '.join([char_lut[xch_i[j]] for j in range(min(char_limit, xch_i.shape[0]))])
            word_repr = word_repr.replace('<PADDING>', '').replace(' ', '')
            print('\t%s' % word_repr)
        print(']')


def show_batch_all(batch, word_lut, char_lut, word_limit=1000, char_limit = 200):

    for x, xch in zip(batch[0], batch[1]):
        print(' '.join([word_lut[x[i]] for i in range(min(word_limit, x.shape[0]))]))
        print('[')
        for i in range(min(word_limit,xch.shape[0])):
            xch_i = xch[i]
            word_repr = ' '.join([char_lut[xch_i[j]] for j in range(min(char_limit, xch_i.shape[0]))])
            word_repr = word_repr.replace('<PADDING>', '').replace(' ', '')
            print('\t%s' % word_repr)
        print(']')


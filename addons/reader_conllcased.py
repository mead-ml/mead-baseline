from baseline.reader import CONLLSeqReader
import numpy as np
from collections import Counter
import codecs
import io
import os
import re
import baseline.data


class CONLLSeqMixedCaseReader(CONLLSeqReader):

    def __init__(self, max_sentence_length=-1, max_word_length=-1, word_trans_fn=None,
                 vec_alloc=np.zeros, vec_shape=np.shape, trim=False, extended_features=dict()):
        super().__init__(max_sentence_length, max_word_length, word_trans_fn, vec_alloc, vec_shape, trim, extended_features)
        self.idx = 2 # GO=0, START=1, EOS=2


    def load(self, filename, vocabs, batchsz, shuffle=False, do_sort=True):
        ts = []
        words_vocab = vocabs['word']
        chars_vocab = vocabs['char']
        mxlen = self.max_sentence_length
        maxw = self.max_word_length
        extracted = self.read_examples(filename)
        texts = extracted['texts']
        labels = extracted['labels']

        for i in range(len(texts)):

            xs_ch = self.vec_alloc((mxlen, maxw), dtype=np.int)
            xs = self.vec_alloc((mxlen), dtype=np.int)
            xs_lc = self.vec_alloc((mxlen), dtype=np.int)
            ys = self.vec_alloc((mxlen), dtype=np.int)
            keys = self.extended_features.keys()

            item = {}
            for key in keys:
                item[key] = self.vec_alloc((mxlen), dtype=np.int)

            text = texts[i]
            lv = labels[i]

            length = mxlen
            for j in range(mxlen):

                if j == len(text):
                    length = j
                    break

                w = text[j]
                nch = min(len(w), maxw)
                label = lv[j]

                if label not in self.label2index:
                    self.idx += 1
                    self.label2index[label] = self.idx

                ys[j] = self.label2index[label]
                xs_lc[j] = words_vocab.get(w.lower(), 0)
                xs[j] = words_vocab.get(w, 0)
                # Extended features
                for key in keys:
                    item[key][j] = vocabs[key].get(extracted[key][i][j])
                for k in range(nch):
                    xs_ch[j, k] = chars_vocab.get(w[k], 0)
            item.update({'x': xs, 'x_lc': xs_lc, 'xch': xs_ch, 'y': ys, 'lengths': length, 'ids': i})
            ts.append(item)
        examples = baseline.data.SeqWordCharTagExamples(ts, do_shuffle=shuffle, do_sort=do_sort)
        return baseline.data.SeqWordCharLabelDataFeed(examples, batchsz=batchsz, shuffle=shuffle,
                                                      vec_alloc=self.vec_alloc, vec_shape=self.vec_shape), texts

    def build_vocab(self, files):
        vocab_word = Counter()
        vocab_ch = Counter()
        vocab_word['<UNK>'] = 1
        vocabs = {}
        keys = self.extended_features.keys()
        for key in keys:
            vocabs[key] = Counter()

        maxw = 0
        maxs = 0
        for file in files:
            if file is None:
                continue

            sl = 0
            with codecs.open(file, encoding='utf-8', mode='r') as f:
                for line in f:

                    line = line.strip()
                    if line == '':
                        maxs = max(maxs, sl)
                        sl = 0

                    else:
                        states = re.split("\s", line)
                        sl += 1
                        w = states[0]
                        vocab_word[w] += 1
                        vocab_word[w.lower()] += 1
                        maxw = max(maxw, len(w))
                        for k in w:
                            vocab_ch[k] += 1
                        for key, index in self.extended_features.items():
                            vocabs[key][states[index]] += 1

        self.max_word_length = min(maxw, self.max_word_length) if self.max_word_length > 0 else maxw
        self.max_sentence_length = min(maxs, self.max_sentence_length) if self.max_sentence_length > 0 else maxs
        print('Max sentence length %d' % self.max_sentence_length)
        print('Max word length %d' % self.max_word_length)

        vocabs.update({'char': vocab_ch, 'word': vocab_word})
        return vocabs


def create_seq_pred_reader(mxlen, mxwlen, word_trans_fn, vec_alloc, vec_shape, trim, **kwargs):
    return CONLLSeqMixedCaseReader(mxlen, mxwlen, word_trans_fn, vec_alloc, vec_shape, trim)

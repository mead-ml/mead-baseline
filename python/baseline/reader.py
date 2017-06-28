import baseline.data
import numpy as np
from collections import Counter
import re
import codecs


def num_lines(filename):
    lines = 0
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        for _ in f:
            lines = lines + 1
    return lines


class TSVSentencePairReader:

    @staticmethod
    def build_vocab(colids, files, clean=False, chars=False):
        vocab = Counter()
        vocab['<PAD>'] = 1
        vocab['<GO>'] = 1
        vocab['<EOS>'] = 1
        for file in files:
            if file is None:
                continue
            with codecs.open(file, encoding='utf-8', mode='r') as f:
                for line in f:
                    cols = re.split("\t", line)
                    for col in colids:
                        text = re.split("\s", cols[col])

                        for w in text:
                            w = w.strip()
                            vocab[w] += 1
        return vocab

    @staticmethod
    def load(tsfile, vocab1, vocab2, mxlen, vec_alloc=np.zeros):

        PAD = vocab1['<PADDING>']
        GO = vocab2['<GO>']
        EOS = vocab2['<EOS>']

        ts = []
        with codecs.open(tsfile, encoding='utf-8', mode='r') as f:
            for line in f:
                splits = re.split("\t", line.strip())
                src = re.split("\s+", splits[0])
                dst = re.split("\s+", splits[1])

                srcl = vec_alloc(mxlen, dtype=np.int)
                tgtl = vec_alloc(mxlen, dtype=np.int)
                src_len = len(src)
                tgt_len = len(dst) + 2
                end1 = min(src_len, mxlen)
                end2 = min(tgt_len, mxlen)-2
                last = max(end1, end2)
                tgtl[0] = GO
                src_len = end1
                tgt_len = end2+2

                for j in range(last):
                    idx1 = vocab1[src[j]] if j < end1 else PAD
                    idx2 = vocab2[dst[j]] if j < end2 else PAD
                    srcl[j] = idx1
                    tgtl[j + 1] = idx2

                tgtl[end2] = EOS

                ts.append( (srcl, tgtl, src_len, tgt_len) )
        return baseline.data.Seq2SeqExamples(ts)


class TSVSeqLabelReader:

    REPLACE = { "'s": " 's ",
                "'ve": " 've ",
                "n't": " n't ",
                "'re": " 're ",
                "'d": " 'd ",
                "'ll": " 'll ",
                ",": " , ",
                "!": " ! ",
                }

    @staticmethod
    def splits(text):
        return list(filter(lambda s: len(s) != 0, re.split('\s+', text)))

    @staticmethod
    def do_clean(l):
        l = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", l)
        for k, v in TSVSeqLabelReader.REPLACE.items():
            l = l.replace(k, v)
        return l.strip()

    @staticmethod
    def label_and_sentence(line, clean, chars):
        labelText = re.split('[\t\s]+', line)
        label = labelText[0]
        text = labelText[1:]
        if chars is True:
            text = ' '.join([ch for ch in ''.join(text)])
        if clean is True:
            text = ' '.join([TSVSeqLabelReader.do_clean(w.lower()) for w in text]).replace('  ', ' ')
        else:
            text = ' '.join(text).replace('  ', ' ')
        return label, text

    @staticmethod
    def build_vocab(files, clean=False, chars=False):
        vocab = Counter()
        for file in files:
            if file is None:
                continue
            with codecs.open(file, encoding='utf-8', mode='r') as f:
                for line in f:
                    _, text = TSVSeqLabelReader.label_and_sentence(line, clean, chars)
                    for w in TSVSeqLabelReader.splits(text):
                        vocab[w] += 1
        return vocab

    @staticmethod
    def load(filename, index, f2i, clean=False, chars=False, mxlen=1000, mxfiltsz=0, vec_alloc=np.zeros):

        PAD = index['<PADDING>']
        halffiltsz = mxfiltsz // 2
        nozplen = mxlen - 2*halffiltsz
        label_idx = len(f2i)
        examples = []
        with codecs.open(filename, encoding='utf-8', mode='r') as f:
            for offset, line in enumerate(f):
                label, text = TSVSeqLabelReader.label_and_sentence(line, clean, chars)
                if label not in f2i:
                    f2i[label] = label_idx
                    label_idx += 1

                y = f2i[label]
                toks = TSVSeqLabelReader.splits(text)
                mx = min(len(toks), nozplen)
                toks = toks[:mx]
                x = vec_alloc(mxlen, dtype=int)
                for j in range(len(toks)):
                    w = toks[j]
                    key = index.get(w, PAD)
                    x[j+halffiltsz] = key
                examples.append((x, y))
        return baseline.data.SeqLabelExamples(examples), f2i


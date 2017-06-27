import w2v
import numpy as np
from collections import Counter
import re
import math
import codecs
import data

def num_lines(filename):
    lines = 0
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        for line in f:
            lines = lines + 1
    return lines

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


def load_sentences(tsfile, vocab1, vocab2, mxlen, vec_alloc=np.zeros):

    PAD = vocab1['<PADDING>']
    GO = vocab2['<GO>']
    EOS = vocab2['<EOS>']

    ts = []
    b = 0
    i = 0
    n = num_lines(tsfile)

    with codecs.open(tsfile, encoding='utf-8', mode='r') as f:
        for line in f:
            splits = re.split("\t", line.strip())
            src = re.split("\s+", splits[0])
            dst = re.split("\s+", splits[1])

            srcl = vec_alloc((mxlen), dtype=np.int)
            tgtl = vec_alloc((mxlen), dtype=np.int)
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
    return data.Examples(ts)

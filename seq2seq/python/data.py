import w2v
import numpy as np
from collections import Counter
import re
import math
import codecs

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


def load_sentences(tsfile, vocab1, vocab2, mxlen, batchsz):

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
            offset = i % batchsz
            if offset == 0:
                if b > 0:
                    ts.append({"src":srcl,"dst":dstl,"tgt":tgtl})
                b = b + 1
                thisBatchSz = min(batchsz, n - i)
                srcl = np.zeros((thisBatchSz, mxlen))
                dstl = np.zeros((thisBatchSz, mxlen))
                tgtl = np.zeros((thisBatchSz, mxlen))


            end1 = min(len(src), mxlen)
            end2 = min(len(dst), mxlen)
            mxsiglen = max(end1, end2)
            dstl[offset, 1] = GO

       
            for j in range(mxsiglen):
                idx1 = j < end1 and vocab1[src[j]] or PAD
                idx2 = j < end2 and vocab2[dst[j]] or PAD
                # First signal is reversed and starting at end, left padding
                srcl[offset, mxlen - (j+1)] = idx1
                # Second signal is not reversed, follows <go> and ends on <eos>
                dstl[offset, j + 1] = idx2
                tgtl[offset, j] = idx2 == PAD and 0 or idx2

            tgtl[offset, end2] = EOS
            i = i + 1

    if thisBatchSz > 0:
       ts.append({"src":srcl,"dst":dstl,"tgt":tgtl})

    return ts

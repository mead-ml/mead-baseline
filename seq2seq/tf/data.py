import w2v
import numpy as np
from collections import Counter
import re
import math
import codecs

def tryGetWordIdx(w2v, word):
   return w2v.vocab[word] or w2v.vocab[word.lower()]

def idxFor(w2v, tok):
   OOV = w2v.vocab['<PADDING>']
   return tryGetWordIdx(w2v, tok) or OOV

def revlut(lut):
    return {v: k for k, v in lut.items()}

def numLines(filename):
    lines = 0
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        for line in f:
            lines = lines + 1
    return lines


def buildVocab(colids, files, clean=False, chars=False):
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


def sentsToIndices(tsfile, embed1, embed2, options):

    linenum = 1
    
    wsz = embed1.dsz
    batchsz = options.get("batchsz", 1)
    mxlen = options.get("mxlen", 40)

    PAD = embed1.vocab['<PADDING>']
    GO = embed2.vocab['<GO>']
    EOS = embed2.vocab['<EOS>']
    ts = []
    b = 0
    i = 0
    n = numLines(tsfile)

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
                idx1 = j < end1 and idxFor(embed1, src[j]) or PAD
                idx2 = j < end2 and idxFor(embed2, dst[j]) or PAD
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

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

            # if length 100, mxsiglen can be at most 99
            end1 = min(src_len, mxlen - 1)
            end2 = min(tgt_len, mxlen - 1)
            last = max(end1, end2)
            tgtl[0] = GO
            src_len = end1
            tgt_len = end2

            #src_len = mxlen #end1
            #tgt_len = mxlen #end2
       
            for j in range(last):
                idx1 = j < end1 and vocab1[src[j]] or PAD
                idx2 = j < end2 - 2 and vocab2[dst[j]] or PAD
                # First signal is reversed and starting at end, left padding
                srcl[mxlen - (j+1)] = idx1
                tgtl[j + 1] = idx2 == PAD and 0 or idx2

            tgtl[end2] = EOS

            i = i + 1

            ts.append({"src":srcl,"tgt":tgtl, "src_len": src_len, "tgt_len": tgt_len})

    return ts

def batch(ts, start, batchsz, vec_alloc=np.zeros, vec_shape=np.shape):
    ex = ts[start]
    sig_len = vec_shape(ex["src"])[0]
    srcs = vec_alloc((batchsz, sig_len), dtype=np.int)
    tgts = vec_alloc((batchsz, sig_len), dtype=np.int)
    src_lens = vec_alloc((batchsz), dtype=np.int)
    tgt_lens = vec_alloc((batchsz), dtype=np.int)
    sz = len(ts)
    idx = start * batchsz
    for i in range(batchsz):
        if idx >= sz: idx = 0
        
        ex = ts[idx]
        srcs[i] = ex["src"]
        tgts[i] = ex["tgt"]
        src_lens[i] = ex["src_len"]
        tgt_lens[i] = ex["tgt_len"]
        idx += 1

    sub_src = np.max(src_lens)
    sub_tgt = np.max(tgt_lens)
    return {"src": srcs, "tgt": tgts, "src_len": src_lens, "tgt_len": tgt_lens }

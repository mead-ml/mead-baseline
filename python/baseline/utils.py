import numpy as np
import re
import six.moves


def listify(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return x
    if x is None:
        return []
    return [x]

def revlut(lut):
    return {v: k for k, v in lut.items()}

def lookup_sentence(rlut, seq, reverse=False, padchar=''):
    s = seq[::-1] if reverse else seq
    return ' '.join([rlut[idx] if rlut[idx] != '<PADDING>' else padchar for idx in s])

# Get a sparse index (dictionary) of top values
# Note: mutates input for efficiency
def topk(k, probs):

    lut = {}
    i = 0

    while i < k:
        idx = np.argmax(probs)
        lut[idx] = probs[idx]
        probs[idx] = 0
        i += 1
    return lut

#  Prune all elements in a large probability distribution below the top K
#  Renormalize the distribution with only top K, and then sample n times out of that

def beam_multinomial(k, probs):
    
    tops = topk(k, probs)
    i = 0
    n = len(tops.keys())
    ary = np.zeros((n))
    idx = []
    for abs_idx,v in tops.iteritems():
        ary[i] = v
        idx.append(abs_idx)
        i += 1

    ary /= np.sum(ary)
    sample_idx = np.argmax(np.random.multinomial(1, ary))
    return idx[sample_idx]

def fill_y(nc, yidx):
    xidx = np.arange(0, yidx.shape[0], 1)
    dense = np.zeros((yidx.shape[0], nc), dtype=int)
    dense[xidx, yidx] = 1
    return dense

def seq_fill_y(nc, yidx):
    batchsz = yidx.shape[0]
    siglen = yidx.shape[1]
    dense = np.zeros((batchsz, siglen, nc), dtype=np.int)
    for i in range(batchsz):
        for j in range(siglen):
            idx = int(yidx[i, j])
            if idx > 0:
                dense[i, j, idx] = 1

    return dense

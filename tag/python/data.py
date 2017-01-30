import w2v
import numpy as np
from collections import Counter
import re
import math
import codecs
from utils import revlut
UNREP_EMOTICONS = (
    ':)',
    ':(((',
    ':D',
    '=)',
    ':-)',
    '=(',
    '(=',
    '=[[',
)

def cleanup(word):
    if word.startswith('http'): return 'URL'
    if word.startswith('@'): return '@@@@'
    if word.startswith('#'): return '####'
    if word == '"': return ','
    if word in UNREP_EMOTICONS: return ';)'
    if word == '<3': return '&lt;3'
    return word

def num_lines(filename):
    lines = 0
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        for line in f:
            lines = lines + 1
    return lines

def conll_build_vocab(files):
    vocab_word = Counter()
    vocab_ch = Counter()
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
                    vocab_word[cleanup(w)] += 1
                    maxw = max(maxw, len(w))
                    for k in w:
                        vocab_ch[k] += 1

    return maxs, maxw, vocab_ch, vocab_word

def conll_lines(tsfile):

    txts = []
    lbls = []
    txt = []
    lbl = []
    
    with codecs.open(tsfile, encoding='utf-8', mode='r') as f:
        for line in f:
            states = re.split("\s", line.strip())

            if len(states) > 1:
                txt.append(states[0])
                lbl.append(states[-1])
            else:
                txts.append(txt)
                lbls.append(lbl)
                txt = []
                lbl = []


    return txts, lbls

def conll_load_sentences(filename, words_vocab, chars_vocab, mxlen, maxw, f2i, vec_alloc=np.zeros):

    b = 0
    ts = []
    idx = 0
    txts, lbls = conll_lines(filename)

    for i in range(len(txts)):

        xs_ch = vec_alloc((mxlen, maxw), dtype=np.int)
        xs = vec_alloc((mxlen), dtype=np.int)
        ys = vec_alloc((mxlen), dtype=np.int)
        
        lv = lbls[i]
        v = txts[i]
        
        length = mxlen
        for j in range(mxlen):
            
            if j == len(v):
                length = j
                break
            
            w = v[j]
            nch = min(len(w), maxw)
            label = lv[j]

            if not label in f2i:
                idx += 1
                f2i[label] = idx

            ys[j] = f2i[label]
            xs[j] = words_vocab.get(cleanup(w))
            for k in range(nch):
                xs_ch[j,k] = chars_vocab.get(w[k], 0)
        ts.append({"x":xs,"y":ys, "xch": xs_ch, "id": i, "length": length })

    return ts, f2i, txts

def valid_split(data, splitfrac):
    train = []
    valid = []
    numinst = len(data)
    heldout = int(math.floor(numinst * (1-splitfrac)))
    return data[1:heldout], data[heldout:]

def batch(ts, start, batchsz, vec_alloc=np.zeros, vec_shape=np.shape):
    ex = ts[start]
    siglen, maxw = vec_shape(ex["xch"])
    xs_ch = vec_alloc((batchsz, siglen, maxw), dtype=np.int)
    xs = vec_alloc((batchsz, siglen), dtype=np.int)
    ys = vec_alloc((batchsz, siglen), dtype=np.int)
    ids = vec_alloc((batchsz), dtype=np.int)
    length = vec_alloc((batchsz), dtype=np.int)
    sz = len(ts)
    idx = start * batchsz
    for i in range(batchsz):
        if idx >= sz: idx = 0
        
        ex = ts[idx]
        xs_ch[i] = ex["xch"]
        xs[i] = ex["x"]
        ys[i] = ex["y"]
        ids[i] = ex["id"]
        length[i] = ex["length"]
        idx += 1
    return {"x": xs, "y": ys, "xch": xs_ch, "length": length, "id": ids }

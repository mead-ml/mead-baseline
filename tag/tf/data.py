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
    return word.lower()

REPLACE = { "'s": " 's ",
            "'ve": " 've ",
            "n't": " n't ",
            "'re": " 're ",
            "'d": " 'd ",
            "'ll": " 'll ",
            ",": " , ",
            "!": " ! ",
        }

def numLines(filename):
    lines = 0
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        for line in f:
            lines = lines + 1
    return lines

def conllBuildVocab(files):
    vocab_word = Counter()
    vocab_ch = Counter()
    maxw = 0

    for file in files:
        if file is None:
            continue
        with codecs.open(file, encoding='utf-8', mode='r') as f:
            for line in f:
                states = re.split("\s", line)
                if len(states) > 0:
                    w = states[0]
                    vocab_word[w] += 1
                    maxw = max(maxw, len(w))
                    for k in w:
                        vocab_ch[k] += 1
    return maxw, vocab_ch, vocab_word

def conllLines(tsfile):

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

def conllSentsToIndices(filename, words, chars, maxw, f2i, options):

    chsz = chars.dsz
    mxlen = options.get("mxlen", 40)

    b = 0
    ts = []
    idx = 0
    txts, lbls = conllLines(filename)

    for i in range(len(txts)):

        xs_ch = np.zeros((mxlen, maxw), dtype=np.int)
        xs = np.zeros((mxlen), dtype=np.int)
        ys = np.zeros((mxlen), dtype=np.int)
        
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
            xs[j] = words.vocab.get(cleanup(w), 0)
            for k in range(nch):
                xs_ch[j,k] = chars.vocab.get(w[k], 0)
        ts.append({"x":xs,"y":ys, "xch": xs_ch, "id": i, "length": length })

    return ts, f2i, txts

def validSplit(data, splitfrac):
    train = []
    valid = []
    numinst = len(data)
    heldout = int(math.floor(numinst * (1-splitfrac)))
    return data[1:heldout], data[heldout:]

def batch(ts, start, batchsz):
    ex = ts[start]
    siglen = ex["x"].shape[0]
    maxw = ex["xch"].shape[1]
    
    xs_ch = np.zeros((batchsz, siglen, maxw), dtype=np.int)
    xs = np.zeros((batchsz, siglen), dtype=np.int)
    ys = np.zeros((batchsz, siglen), dtype=np.int)
    ids = np.zeros((batchsz), dtype=np.int)
    length = np.zeros((batchsz), dtype=np.int)
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
    return {"x": xs, "y": ys, "xch": xs_ch, "length": length }
        



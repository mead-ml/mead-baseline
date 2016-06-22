import w2v
import numpy as np
from collections import Counter
import re
import math
import codecs

def revlut(lut):
    return {v: k for k, v in lut.items()}

REPLACE = { "'s": " 's ",
            "'ve": " 've ",
            "n't": " n't ",
            "'re": " 're ",
            "'d": " 'd ",
            "'ll": " 'll ",
            ",": " , ",
            "!": " ! ",
        }
          
  
def doClean(l):
    l = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", l)
    for k,v in REPLACE.items():
        l = l.replace(k, v)
    return l.strip()

def labelSent(line, clean, chars):
    labelText = re.split('[\t\s]+', line)
    label = labelText[0]
    text = labelText[1:]
    if chars is True:
        text = ' '.join([ch for ch in ''.join(text)])
    if clean is True:
        text = ' '.join([doClean(w.lower()) for w in text]).replace('  ', ' ')
    else:
        text = ' '.join(text).replace('  ', ' ')
    return label, text

def numLines(filename):
    lines = 0
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        for line in f:
            lines = lines + 1
    return lines

def conllBuildVocab(files):
    vocab = Counter()
    
    for file in files:
        if file is None:
            continue
        with codecs.open(file, encoding='utf-8', mode='r') as f:
            for line in f:
                states = re.split("\s", line)
                if len(states) > 0:
                    w = states[0]
                    vocab[w] += 1
    return vocab

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


def conllSentsToVectors(filename, w2v, f2i, options):
    
    batchsz = options.get("batchsz", 1)
    mxlen = options.get("mxlen", 40)
    zp = options.get("zp", 0)
    b = 0
    ts = []
    idx = 0
    txts, lbls = conllLines(filename)
    for i in range(len(txts)):
        lv = lbls[i]
        v = txts[i]

        offset = i % batchsz
        
        siglen = mxlen + 2*zp
        
        if offset == 0:
            # Write batch
            if b > 0:
                ts.append({"x":xs,"y":ys})
            b += 1
            thisBatchSz = min(batchsz, len(txts) - i)
            xs = np.zeros((thisBatchSz, siglen, wsz))
            ys = np.zeros((thisBatchSz, siglen), dtype=int)

        
        for j in range(min(siglen, len(v))):
            tok = v[j]
            label = lv[j]
            if not label in f2i:
                idx += 1
                f2i[label] = idx
            ys[offset][j+zp] = f2i[label]
#            print(f2i[label])
            xs[offset][j+zp] = w2v.lookup(tok)
    if thisBatchSz > 0:
        ts.append({"x":xs,"y":ys})

    return ts, f2i

def conllSentsToIndices(filename, w2v, f2i, options):
    
    batchsz = options.get("batchsz", 1)
    mxlen = options.get("mxlen", 40)
    zp = options.get("zp", 0)
    b = 0
    ts = []
    idx = 0
    txts, lbls = conllLines(filename)

    for i in range(len(txts)):
        lv = lbls[i]
        v = txts[i]
        offset = i % batchsz
        siglen = mxlen + 2*zp
        
        if offset == 0:
            # Write batch
            if b > 0:
                ts.append({"x":xs,"y":ys})
            b += 1
            thisBatchSz = min(batchsz, len(txts) - i)
            xs = np.zeros((thisBatchSz, siglen))
            ys = np.zeros((thisBatchSz, siglen))

        
        for j in range(min(siglen, len(v))):
            tok = v[j]
            label = lv[j]
            if not label in f2i:
                idx += 1
                f2i[label] = idx
            ys[offset][j+zp] = f2i[label]
            xs[offset][j+zp] = w2v.vocab.get(tok, 0)
    if thisBatchSz > 0:
        ts.append({"x":xs,"y":ys})

    return ts, f2i

def validSplit(data, splitfrac):
    train = []
    valid = []
    numinst = len(data)
    heldout = int(math.floor(numinst * (1-splitfrac)))
    return data[1:heldout], data[heldout:]

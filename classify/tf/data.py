import w2v
import numpy as np
from collections import Counter
import re
import math

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

def labelSent(line, clean):
    labelText = re.split('[\t\s]+', line)
    label = labelText[0]
    text = labelText[1:]
    if clean is True:
        text = ' '.join([doClean(w.lower()) for w in text]).replace('  ', ' ')
    return label, text

def numLines(filename):
    lines = 0
    with open(filename, "r") as f:
        for line in f:
            lines = lines + 1
    return lines

def buildVocab(files, clean=False):
    vocab = Counter()
    for file in files:
        if file is None:
            continue
        with open(file, "r") as f:
            for line in f:
                _, text = labelSent(line, clean)
                for w in re.split("\s", text):
                    vocab[w] += 1
    return vocab

def validSplit(data, splitfrac):
    train = []
    valid = []
    numinst = len(data)
    heldout = int(math.floor(numinst * (1-splitfrac)))
    return data[1:heldout], data[heldout:]


def loadTemporalIndices(filename, w2v, f2i, options):
    ts = []
    batchsz = options.get("batchsz", 1)
    clean = options.get("clean", True)
    vsz = w2v.vsz
    dsz = w2v.dsz
    PAD = w2v.vocab['<PADDING>']
    mxfiltsz = np.max(options["filtsz"])
    mxlen = options.get("mxlen", 1000)
    mxsiglen = mxlen - mxfiltsz
    labelIdx = len(f2i)
    
    halffiltsz = int(math.floor(mxfiltsz / 2))
    labelIdx = len(f2i)

    n = numLines(filename)
    x = y = None
    b = i = 0
    thisBatchSz = 0
    with open(filename, "r") as f:

        for line in f:
            label, text = labelSent(line, clean)
            if not label in f2i:
                f2i[label] = labelIdx
                labelIdx += 1

            offset = i % batchsz

            if offset == 0:
                if b > 0:
                    ts.append({"x":x, "y":y})
                b = b + 1
                thisBatchSz = min(batchsz, n - i)
                x = np.empty((thisBatchSz, mxlen), dtype=np.float32)
                x.fill(PAD)
                y = np.zeros((thisBatchSz), dtype=int)
 
            y[offset] = f2i[label]
            toks = re.split("\s+", text)
            mx = min(len(toks), mxsiglen)
            toks = toks[:mx]
            for j in range(len(toks)):
                w = toks[j]
                key = w2v.vocab.get(w, PAD)
                x[offset][j+halffiltsz] = key
            i = i + 1
    if thisBatchSz > 0:
        ts.append({"x":x,"y":y})
    return ts, f2i    

def loadTemporalEmb(filename, w2v, f2i, options):
    ts = []
    batchsz = options.get("batchsz", 1)
    clean = options.get("clean", True)
    vsz = w2v.vsz
    dsz = w2v.dsz
        
    mxfiltsz = np.max(options["filtsz"])
    mxlen = options.get("mxlen", 1000)
    mxsiglen = mxlen - mxfiltsz
    halffiltsz = int(math.floor(mxfiltsz / 2))
    labelIdx = len(f2i)

    n = numLines(filename)
    x = y = None
    b = i = 0
    thisBatchSz = 0
    with open(filename, "r") as f:

        for line in f:
            label, text = labelSent(line, clean)
            if not label in f2i:
                f2i[label] = labelIdx
                labelIdx += 1

            offset = i % batchsz

            if offset == 0:
                if b > 0:
                    ts.append({"x":x, "y":y})
                b = b + 1
                thisBatchSz = min(batchsz, n - i)
                x = np.zeros((thisBatchSz, mxlen, dsz), dtype=np.float32)
                y = np.zeros((thisBatchSz), dtype=int)
            y[offset] = f2i[label]
            toks = re.split("\s+", text)
            mx = min(len(toks), mxsiglen)
            toks = toks[:mx]
            for j in range(len(toks)):
                w = toks[j]
                z = w2v.lookup(w, False)
#                print(z)
                x[offset][j + halffiltsz] = z
            i = i + 1
    if thisBatchSz > 0:
        ts.append({"x":x,"y":y})
    return ts, f2i

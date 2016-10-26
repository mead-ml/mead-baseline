import w2v
import numpy as np
from collections import Counter
import re
import math
import codecs

REPLACE = { "'s": " 's ",
            "'ve": " 've ",
            "n't": " n't ",
            "'re": " 're ",
            "'d": " 'd ",
            "'ll": " 'll ",
            ",": " , ",
            "!": " ! ",
        }
          
  
def splits(text):
    return filter(lambda s: len(s) != 0, re.split('\s+', text))

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

def buildVocab(files, clean=False, chars=False):
    vocab = Counter()
    for file in files:
        if file is None:
            continue
        with codecs.open(file, encoding='utf-8', mode='r') as f:
            for line in f:
                _, text = labelSent(line, clean, chars)
                for w in splits(text):
                    vocab[w] += 1
    return vocab

def validSplit(data, splitfrac):
    train = []
    valid = []
    numinst = len(data)
    heldout = int(math.floor(numinst * (1-splitfrac)))
    return data[1:heldout], data[heldout:]


def loadTemporalIndices(filename, index, f2i, options):
    ts = []
    batchsz = options.get("batchsz", 1)
    clean = options.get("clean", True)
    chars = options.get("chars", False)
    PAD = index['<PADDING>']
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

    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        for line in f:
            label, text = labelSent(line, clean, chars)
            if not label in f2i:
                f2i[label] = labelIdx
                labelIdx += 1

            offset = i % batchsz

            if offset == 0:
                if b > 0:
                    ts.append({"x":x, "y":y})
                b = b + 1
                thisBatchSz = min(batchsz, n - i)
                x = np.empty((thisBatchSz, mxlen), dtype=int)
                x.fill(PAD)
                y = np.zeros((thisBatchSz), dtype=int)
 
            y[offset] = f2i[label]
            toks = splits(text)
            mx = min(len(toks), mxsiglen)
            toks = toks[:mx]
            for j in range(len(toks)):
                w = toks[j]
                key = index.get(w, PAD)
                x[offset][j+halffiltsz] = key
            i = i + 1
    if thisBatchSz > 0:
        ts.append({"x":x,"y":y})
    return ts, f2i    

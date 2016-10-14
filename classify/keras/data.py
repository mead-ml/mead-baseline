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

def loadTemporalIndices(filename, index, f2i, clean=False, chars=False, mxlen=1000, mxfiltsz=5):
    ts = []
    PAD = index['<PADDING>']
    mxsiglen = mxlen - mxfiltsz
    labelIdx = len(f2i)
    
    halffiltsz = int(math.floor(mxfiltsz / 2))
    labelIdx = len(f2i)

    n = numLines(filename)
    x = np.empty((n, mxlen), dtype=int)
    x.fill(PAD)
    y = np.zeros(n, dtype=int)
 
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        for offset, line in enumerate(f):
            label, text = labelSent(line, clean, chars)
            if not label in f2i:
                f2i[label] = labelIdx
                labelIdx += 1

            y[offset] = f2i[label]
            toks = splits(text)
            mx = min(len(toks), mxsiglen)
            toks = toks[:mx]
            for j in range(len(toks)):
                w = toks[j]
                key = index.get(w, PAD)
                x[offset][j+halffiltsz] = key
    return x, y, f2i    


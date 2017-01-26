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
    return list(filter(lambda s: len(s) != 0, re.split('\s+', text)))

def do_clean(l):
    l = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", l)
    for k,v in REPLACE.items():
        l = l.replace(k, v)
    return l.strip()

def label_and_sentence(line, clean, chars):
    labelText = re.split('[\t\s]+', line)
    label = labelText[0]
    text = labelText[1:]
    if chars is True:
        text = ' '.join([ch for ch in ''.join(text)])
    if clean is True:
        text = ' '.join([do_clean(w.lower()) for w in text]).replace('  ', ' ')
    else:
        text = ' '.join(text).replace('  ', ' ')
    return label, text

def num_lines(filename):
    lines = 0
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        for line in f:
            lines = lines + 1
    return lines

def build_vocab(files, clean=False, chars=False):
    vocab = Counter()
    for file in files:
        if file is None:
            continue
        with codecs.open(file, encoding='utf-8', mode='r') as f:
            for line in f:
                _, text = label_and_sentence(line, clean, chars)
                for w in splits(text):
                    vocab[w] += 1
    return vocab

class SentenceLabelExamples(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def width(self):
        return self.x[0].shape[0]

def valid_split(data, splitfrac, ExType=SentenceLabelExamples):
    numinst = len(data)
    heldout = int(math.floor(numinst * (1-splitfrac)))
    x = data.x
    y = data.y
    return ExType(x[1:heldout], y[1:heldout]), ExType(x[heldout:], y[heldout:])


def load_sentences(filename, index, f2i, clean=False, chars=False, mxlen=1000, mxfiltsz=0, vec_alloc=np.zeros, ExType=SentenceLabelExamples):
    ts = []
    PAD = index['<PADDING>']

    labelIdx = len(f2i)
    
    halffiltsz = int(math.floor(mxfiltsz / 2))
    nozplen = mxlen - 2*halffiltsz
    labelIdx = len(f2i)

    n = num_lines(filename)
    x = vec_alloc((n, mxlen), dtype=int)
    y = vec_alloc((n), dtype=int)
 
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        for offset, line in enumerate(f):
            label, text = label_and_sentence(line, clean, chars)
            if not label in f2i:
                f2i[label] = labelIdx
                labelIdx += 1

            y[offset] = f2i[label]
            toks = splits(text)
            mx = min(len(toks), nozplen)
            toks = toks[:mx]
            for j in range(len(toks)):
                w = toks[j]
                key = index.get(w, PAD)
                x[offset][j+halffiltsz] = key
    return ExType(x, y), f2i    


def batch(dataset, start, batchsz, vec_alloc=np.empty, ExType=SentenceLabelExamples):
    siglen = dataset.width()
    xb = vec_alloc((batchsz, siglen), dtype=np.int)

    yb = vec_alloc((batchsz), dtype=np.int)
    sz = len(dataset)
    idx = start * batchsz
    for i in range(batchsz):
        if idx >= sz: idx = 0
        x, y = dataset[idx]

        xb[i] = x
        yb[i] = y
        idx += 1

    return ExType(xb, yb)

import numpy as np
import time
import os
import importlib


def listify(x):
    """Take a scalar or list and make it a list
    
    :param x: The input to convert
    :return: A list
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return x
    if x is None:
        return []
    return [x]


def revlut(lut):
    return {v: k for k, v in lut.items()}


def import_user_module(module_type, model_type):
    module_name = "%s_%s" % (module_type, model_type)
    print('Loading user model %s' % module_name)
    mod = importlib.import_module(module_name)
    return mod


def create_user_model(w2v, labels, **kwargs):
    model_type = kwargs['model_type']
    mod = import_user_module("classifier", model_type)
    return mod.create_model(w2v, labels, **kwargs)


def load_user_model(outname, **kwargs):
    model_type = kwargs['model_type']
    mod = import_user_module("classifier", model_type)
    return mod.load_model(model_type, outname, **kwargs)


def create_user_tagger_model(labels, word_embedding, char_embedding, **kwargs):
    model_type = kwargs['model_type']
    mod = import_user_module("tagger", model_type)
    return mod.create_model(labels, word_embedding, char_embedding, **kwargs)


def load_user_tagger_model(outname, **kwargs):
    model_type = kwargs['model_type']
    mod = import_user_module("tagger", model_type)
    return mod.load_model(model_type, outname, **kwargs)


def get_model_file(dictionary, task, platform):
    base = dictionary.get('outfile', './%s-model' % task)
    rid = os.getpid()
    if platform.startswith('pyt'):
        name = '%s-%d.pyt' % (base, rid)
    else:
        name = '%s-%s-%d' % (base, platform, rid)
    print('model file [%s]' % name)
    return name


def lookup_sentence(rlut, seq, reverse=False, padchar=''):
    s = seq[::-1] if reverse else seq
    return (' '.join([rlut[idx] if rlut[idx] != '<PAD>' else padchar for idx in s])).strip()


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
    """Convert a `B` sparse array to a dense one, to expand labels 
    
    :param nc: (``int``) The number of labels
    :param yidx: The sparse array of the labels
    :return: A dense array
    """
    xidx = np.arange(0, yidx.shape[0], 1)
    dense = np.zeros((yidx.shape[0], nc), dtype=int)
    dense[xidx, yidx] = 1
    return dense


def seq_fill_y(nc, yidx):
    """Convert a `BxT` sparse array to a dense one, to expand labels 
    
    :param nc: (``int``) The number of labels
    :param yidx: The sparse array of the labels
    :return: A dense array
    """
    batchsz = yidx.shape[0]
    siglen = yidx.shape[1]
    dense = np.zeros((batchsz, siglen, nc), dtype=np.int)
    for i in range(batchsz):
        for j in range(siglen):
            idx = int(yidx[i, j])
            if idx > 0:
                dense[i, j, idx] = 1

    return dense


# Turn a sequence of IOB chunks into single tokens
def to_spans(sequence, lut, strict_iob2=False):

    iobtype = 2 if strict_iob2 else 1
    chunks = []
    current = None

    for i, y in enumerate(sequence):
        label = lut[y]

        #if label.startswith('B-'):
        if not label.startswith('I-') and not label == 'O':
            if current is not None:
                chunks.append('@'.join(current))
            current = [label.replace('B-', ''), '%d' % i ]

        elif label.startswith('I-'):

            if current is not None:
                base = label.replace('I-', '')
                if base == current[0]:
                    current.append('%d' % i)
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print('Warning, type=IOB2, unexpected format ([%s] follows other tag type [%s] @ %d)' % (label, current[0], i))

                    current = [base, '%d' % i]

            else:
                current = [label.replace('I-', ''), '%d' % i]
                if iobtype == 2:
                    print('Warning, unexpected format (I before B @ %d) %s' % (i, label))
        else:
            if current is not None:
                chunks.append('@'.join(current))
            current = None

    if current is not None:
        chunks.append('@'.join(current))

    return set(chunks)


def f_score(overlap_count, gold_count, guess_count, f=1):
    beta_sq = f*f
    if guess_count == 0: return 0.0
    precision = overlap_count / float(guess_count)
    recall = overlap_count / float(gold_count)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    f = (1. + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    return f

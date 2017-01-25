import json
import numpy as np


# We need to keep around our vector maps to preserve lookups of words
def mdsave(labels, vocab, outdir, save_base):
    basename = '%s/%s' % (outdir, save_base)
    
    label_file = basename + '.labels'
    print("Saving attested labels '%s'" % label_file)
    with open(label_file, 'w') as f:
        json.dump(labels, f)

    vocab_file = basename + '.vocab'
    print("Saving attested vocabulary '%s'" % vocab_file)
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f)

def revlut(lut):
    return {v: k for k, v in lut.items()}

def fill_y(nc, yidx):
    xidx = np.arange(0, yidx.shape[0], 1)
    dense = np.zeros((yidx.shape[0], nc), dtype=int)
    dense[xidx, yidx] = 1
    return dense

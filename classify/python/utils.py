import json
import numpy as np
import re
import six.moves

# If you have scipy, use that instead
class ConfusionMatrix:

    def __init__(self, labels):
        #np.set_printoptions(linewidth=300)
        if type(labels) is dict:
            self.labels = []
            for i in range(len(labels)):
                self.labels.append(labels[i])
        else:
            self.labels = labels
        nc = len(self.labels)
        self._cm = np.zeros((nc, nc), dtype=np.int)

    def add(self, truth, guess):
        self._cm[truth, guess] += 1

    def __str__(self):
        values = []
        width = max(5, max(len(x) for x in self.labels) + 1)
        for i, label in enumerate([''] + self.labels):
            values += ["{:>{width}}".format(label, width=width+1)]
        values += ['\n']
        for i, label in enumerate(self.labels):
            values += ["{:>{width}}".format(label, width=width+1)]
            for j in range(len(self.labels)):
                values += ["{:{width}d}".format(self._cm[i, j], width=width + 1)]
            values += ['\n']
        values += ['\n']
        return ''.join(values)

    def reset(self):
        self._cm *= 0

    def get_correct(self):
        return self._cm.diagonal().sum()

    def get_total(self):
        return self._cm.sum()

    def get_acc(self):
        return self.get_correct()/self.get_total()

    def add_batch(self, truth, guess):
        for truth_i, guess_i in zip(truth, guess):
            self.add(truth_i, guess_i)


# Modifed from here
# http://stackoverflow.com/questions/3160699/python-progress-bar#3160819
class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='='):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def update(self, step=1):
        self.current += step
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        six.print_('\r' + self.fmt % args, end='')

    def done(self):
        self.current = self.total
        self.update(step=0)
        print('')



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

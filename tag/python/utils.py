import numpy as np
import csv
import codecs
import cStringIO

def revlut(lut):
    return {v: k for k, v in lut.items()}

# Turn a sequence of IOB chunks into single tokens
def to_spans(sequence, lut, strict_iob2=False):

    iobtype = 2 if strict_iob2 else 1
    chunks = []
    current = None

    for i, y in enumerate(sequence):
        label = lut[y]

        if label.startswith('B-'):
            if current is not None:
                chunks.append('@'.join(current))
            current = [ label.replace('B-', ''), '%d' % i ]

        elif label.startswith('I-'):
            
            if current is not None:
                base = label.replace('I-', '')
                if base == current[0]:
                    current.append('%d' % i)
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print('Warning, type=IOB2, unexpected format ([%s] follows other tag type [%s] @ %d)' % (label, current[0], i))

                    current = [ base, '%d' % i]

            else:
                current = [ label.replace('I-', ''), '%d' % i]
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
    #print('precision %.2f, recall %.2f' % (precision*100., recall*100.))
    if precision == 0.0 or recall == 0.0:
        return 0.0
    f = (1. + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    return f

def fill_y(nc, yidx):
    batchsz = yidx.shape[0]
    siglen = yidx.shape[1]
    dense = np.zeros((batchsz, siglen, nc), dtype=np.int)
    for i in range(batchsz):
        for j in range(siglen):
            idx = int(yidx[i, j])
            if idx > 0:
                dense[i, j, idx] = 1

    return dense


# (B, T, L), gets a one out of L at each T if its populated
# Then get a sum of the populated values
def sentence_lengths(yfilled):
    used = tf.sign(tf.reduce_max(tf.abs(yfilled), reduction_indices=2))
    lengths = tf.reduce_sum(used, reduction_indices=1)
    lengths = tf.cast(lengths, tf.int32)
    total = tf.reduce_sum(lengths)
    #return length
    return total

class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)

def write_embeddings_tsv(word_vec, filename):
    idx2word = revlut(word_vec.vocab)
    with codecs.open(filename, 'w') as f:
        wrtr = UnicodeWriter(f, delimiter='\t', quotechar='"')
#        wrtr.writerow(['Word'])
        for i in range(len(idx2word)):
            row = idx2word[i]
            wrtr.writerow([row])

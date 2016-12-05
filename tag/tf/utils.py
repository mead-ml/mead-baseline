import tensorflow as tf
import numpy as np
import csv
import codecs
import cStringIO

def tensorToSeq(tensor):
    return tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))

def seqToTensor(sequence):
    return tf.transpose(tf.pack(sequence), perm=[1, 0, 2])

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
def sentenceLengths(yfilled):
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

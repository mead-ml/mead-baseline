import random
import numpy as np
import math


class DataFeed:

    def __init__(self, examples, batchsz, **kwargs):
        self.examples = examples
        self.batchsz = batchsz
        self.shuffle = kwargs['shuffle'] if 'shuffle' in kwargs else False
        self.vec_alloc = kwargs['alloc_fn'] if 'alloc_fn' in kwargs else np.zeros
        self.vec_shape = kwargs['shape_fn'] if 'shape_fn' in kwargs else np.shape
        self.src_vec_trans = kwargs['src_trans_fn'] if 'src_trans_fn' in kwargs else None
        self.steps = int(math.floor(len(self.examples)/float(batchsz)))
        self.trim = bool(kwargs['trim']) if 'trim' in kwargs  else False

    def _batch(self, i):
        pass

    def __getitem__(self, i):
        return self._batch(i)

    def __iter__(self):
        shuffle = np.random.permutation(np.arange(self.steps)) if self.shuffle else np.arange(self.steps)

        for i in range(self.steps):
            si = shuffle[i]
            yield self._batch(si)

    def __len__(self):
        return self.steps

class SeqLabelExamples(object):

    SEQ = 0
    LABEL = 1
    def __init__(self, example_list, do_shuffle=True):
        self.example_list = example_list
        if do_shuffle:
            random.shuffle(self.example_list)

    def __getitem__(self, i):
        ex = self.example_list[i]
        return ex[SeqLabelExamples.SEQ], ex[SeqLabelExamples.LABEL]
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.example_list)

    def width(self):
        x, y = self.example_list[0]
        return len(x)


    def batch(self, start, batchsz, vec_alloc=np.empty):
        siglen = self.width()
        xb = vec_alloc((batchsz, siglen), dtype=np.int)
        yb = vec_alloc((batchsz), dtype=np.int)
        sz = len(self.example_list)
        idx = start * batchsz
        for i in range(batchsz):
            if idx >= sz: idx = 0
            x, y = self.example_list[idx]
            xb[i] = x
            yb[i] = y
            idx += 1

        return xb, yb
        
    @staticmethod
    def valid_split(data, splitfrac=0.15):
        numinst = len(data.examples)
        heldout = int(math.floor(numinst * (1-splitfrac)))
        heldout_ex = self.example_list[1:heldout]
        rest_ex = self.example_list[heldout:]
        return SeqLabelExamples(heldout_ex), SeqLabelExamples(rest_ex)

class SeqLabelDataFeed(DataFeed):

    def _batch(self, i):
        x, y = self.examples.batch(i, self.batchsz, self.vec_alloc)
        if self.src_vec_trans is not None:
            x = self.src_vec_trans(x)
        return x, y


class Seq2SeqExamples:

    SRC = 0
    TGT = 1
    SRC_LEN = 2
    TGT_LEN = 3

    def __init__(self, example_list, do_shuffle=True, do_sort=True):
        self.example_list = example_list
        if do_shuffle:
            random.shuffle(self.example_list)
        if do_sort:
            self.example_list = sorted(self.example_list, key=lambda x: x[Seq2SeqExamples.SRC_LEN])

    def __getitem__(self, i):
        ex = self.example_list[i]
        return ex[Seq2SeqExamples.SRC], ex[Seq2SeqExamples.TGT], ex[Seq2SeqExamples.SRC_LEN], ex[Seq2SeqExamples.TGT_LEN]

    def __len__(self):
        return len(self.example_list)

    def batch(self, start, batchsz, trim=False, vec_alloc=np.zeros, vec_shape=np.shape):
        sig_src_len = len(self.example_list[0][Seq2SeqExamples.SRC])
        sig_tgt_len = len(self.example_list[0][Seq2SeqExamples.TGT])

        srcs = vec_alloc((batchsz, sig_src_len), dtype=np.int)
        tgts = vec_alloc((batchsz, sig_tgt_len), dtype=np.int)
        src_lens = vec_alloc((batchsz), dtype=np.int)
        tgt_lens = vec_alloc((batchsz), dtype=np.int)
        sz = len(self.example_list)
        idx = start * batchsz
        
        max_src_len = 0
        max_tgt_len = 0

        idx = start * batchsz
        for i in range(batchsz):
            if idx >= sz: idx = 0
        
            example = self.example_list[idx]
            srcs[i] = example[Seq2SeqExamples.SRC]
            tgts[i] = example[Seq2SeqExamples.TGT]
            src_lens[i] = example[Seq2SeqExamples.SRC_LEN]
            tgt_lens[i] = example[Seq2SeqExamples.TGT_LEN]
            max_src_len = max(max_src_len, src_lens[i])
            max_tgt_len = max(max_tgt_len, tgt_lens[i])
            idx += 1

        if trim:
            srcs = srcs[:,0:max_src_len]
            tgts = tgts[:,0:max_tgt_len]

        return srcs, tgts, src_lens, tgt_lens

def reverse_2nd(vec):
    return vec[:,::-1]

class Seq2SeqDataFeed(DataFeed):

    def _batch(self, i):
        src, tgt, src_len, tgt_len = self.examples.batch(i, self.batchsz, self.trim, self.vec_alloc, self.vec_shape)
        if self.src_vec_trans is not None:
            src = self.src_vec_trans(src)
        return src, tgt, src_len, tgt_len

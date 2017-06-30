import random
import numpy as np
import math


class DataFeed(object):

    def __init__(self):
        self.steps = 0
        self.shuffle = False

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


class ExampleDataFeed(DataFeed):

    def __init__(self, examples, batchsz, **kwargs):
        super(ExampleDataFeed, self).__init__()

        self.examples = examples
        self.batchsz = batchsz
        self.shuffle = kwargs['shuffle'] if 'shuffle' in kwargs else False
        self.vec_alloc = kwargs['alloc_fn'] if 'alloc_fn' in kwargs else np.zeros
        self.vec_shape = kwargs['shape_fn'] if 'shape_fn' in kwargs else np.shape
        self.src_vec_trans = kwargs['src_trans_fn'] if 'src_trans_fn' in kwargs else None
        self.steps = int(math.floor(len(self.examples)/float(batchsz)))
        self.trim = bool(kwargs['trim']) if 'trim' in kwargs  else False



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
        heldout_ex = data.example_list[1:heldout]
        rest_ex = data.example_list[heldout:]
        return SeqLabelExamples(heldout_ex), SeqLabelExamples(rest_ex)

class SeqLabelDataFeed(ExampleDataFeed):

    def __init__(self, examples, batchsz, **kwargs):
        super(SeqLabelDataFeed, self).__init__(examples, batchsz, **kwargs)

    def _batch(self, i):
        x, y = self.examples.batch(i, self.batchsz, self.vec_alloc)
        if self.src_vec_trans is not None:
            x = self.src_vec_trans(x)
        return x, y

class SeqWordCharTagExamples(object):

    SEQ_WORD = 0
    SEQ_CHAR = 1
    SEQ_TAG = 2
    SEQ_LEN = 3
    SEQ_ID = 4

    def __init__(self, example_list, do_shuffle=True, do_sort=True):
        self.example_list = example_list
        if do_shuffle:
            random.shuffle(self.example_list)
        if do_sort:
            self.example_list = sorted(self.example_list, key=lambda x: x[SeqWordCharTagExamples.SEQ_LEN])

    def __getitem__(self, i):
        ex = self.example_list[i]
        return ex[SeqWordCharTagExamples.SEQ_WORD], ex[SeqWordCharTagExamples.SEQ_CHAR], \
               ex[SeqWordCharTagExamples.SEQ_TAG], ex[SeqWordCharTagExamples.SEQ_LEN], \
               ex[SeqWordCharTagExamples.SEQ_ID]

    def __len__(self):
        return len(self.example_list)

    def batch(self, start, batchsz, trim=False, vec_alloc=np.empty, vec_shape=np.shape):
        ex = self.example_list[start]
        siglen, maxw = vec_shape(ex[SeqWordCharTagExamples.SEQ_CHAR])
        xs_ch = vec_alloc((batchsz, siglen, maxw), dtype=np.int)
        xs = vec_alloc((batchsz, siglen), dtype=np.int)
        ys = vec_alloc((batchsz, siglen), dtype=np.int)
        ids = vec_alloc((batchsz), dtype=np.int)
        length = vec_alloc((batchsz), dtype=np.int)
        sz = len(self.example_list)
        idx = start * batchsz

        max_src_len = 0

        for i in range(batchsz):
            if idx >= sz: idx = 0

            ex = self.example_list[idx]
            xs[i] = ex[SeqWordCharTagExamples.SEQ_WORD]
            xs_ch[i] = ex[SeqWordCharTagExamples.SEQ_CHAR]
            ys[i] = ex[SeqWordCharTagExamples.SEQ_TAG]
            length[i] = ex[SeqWordCharTagExamples.SEQ_LEN]
            max_src_len = max(max_src_len, length[i])
            ids[i] = ex[SeqWordCharTagExamples.SEQ_ID]
            idx += 1

        if trim:
            xs = xs[:,0:max_src_len]
            xs_ch = xs_ch[:,0:max_src_len,:]
            ys = ys[:,0:max_src_len]

        return xs, xs_ch, ys, length, ids


    @staticmethod
    def valid_split(data, splitfrac=0.15):
        numinst = len(data.examples)
        heldout = int(math.floor(numinst * (1-splitfrac)))
        heldout_ex = data.example_list[1:heldout]
        rest_ex = data.example_list[heldout:]
        return SeqLabelExamples(heldout_ex), SeqLabelExamples(rest_ex)


class SeqWordCharLabelDataFeed(ExampleDataFeed):

    def __init__(self, examples, batchsz, **kwargs):
        super(SeqWordCharLabelDataFeed, self).__init__(examples, batchsz, **kwargs)

    def _batch(self, i):
        return self.examples.batch(i, self.batchsz, self.trim, self.vec_alloc, self.vec_shape)


class Seq2SeqExamples(object):

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

    def batch(self, start, batchsz, trim=False, vec_alloc=np.zeros):
        sig_src_len = len(self.example_list[0][Seq2SeqExamples.SRC])
        sig_tgt_len = len(self.example_list[0][Seq2SeqExamples.TGT])

        srcs = vec_alloc((batchsz, sig_src_len), dtype=np.int)
        tgts = vec_alloc((batchsz, sig_tgt_len), dtype=np.int)
        src_lens = vec_alloc((batchsz), dtype=np.int)
        tgt_lens = vec_alloc((batchsz), dtype=np.int)
        sz = len(self.example_list)

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
    return vec[:, ::-1]


class Seq2SeqDataFeed(ExampleDataFeed):

    def __init__(self, examples, batchsz, **kwargs):
        super(Seq2SeqDataFeed, self).__init__(examples, batchsz, **kwargs)

    def _batch(self, i):
        src, tgt, src_len, tgt_len = self.examples.batch(i, self.batchsz, self.trim, self.vec_alloc)
        if self.src_vec_trans is not None:
            src = self.src_vec_trans(src)
        return src, tgt, src_len, tgt_len


# This one is a little different at the moment
class SeqWordCharDataFeed(DataFeed):

    def __init__(self, x, xch, nbptt, batchsz, maxw):
        super(SeqWordCharDataFeed, self).__init__()
        num_examples = x.shape[0]
        rest = num_examples // batchsz
        self.steps = rest // nbptt
        self.stride_ch = nbptt * maxw
        trunc = batchsz * rest

        print('Truncating from %d to %d' % (num_examples, trunc))
        self.x = x[:trunc].reshape((batchsz, rest))
        xch = xch.flatten()
        trunc = batchsz * rest * maxw

        print('Truncated from %d to %d' % (xch.shape[0], trunc))
        self.xch = xch[:trunc].reshape((batchsz, rest * maxw))
        self.nbptt = nbptt
        self.batchsz = batchsz
        self.wsz = maxw
    def _batch(self, i):
        return self.x[:, i*self.nbptt:(i+1)*self.nbptt].reshape((self.batchsz, self.nbptt)), \
              self.xch[:, i*self.stride_ch:(i+1)*self.stride_ch].reshape((self.batchsz, self.nbptt, self.wsz)), \
              self.x[:, i*self.nbptt+1:(i+1)*self.nbptt+1].reshape((self.batchsz, self.nbptt))



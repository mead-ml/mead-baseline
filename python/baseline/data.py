import random
import numpy as np
import math


class DataFeed(object):
    """Data collection that, when iterated, produces an epoch of data
    
    This class manages producing a dataset to the trainer, by iterating an epoch and producing
    a single step at a time.  The data can be shuffled per epoch, if requested, otherwise it is 
    returned in the order of the dateset
    """
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

    """Abstract base class that works on a list of examples
    
    """
    def __init__(self, examples, batchsz, **kwargs):
        """Constructor from a list of examples
        
        Use the examples requested to provide data.  Options for batching and shuffling are supported,
        along with some optional processing function pointers
        
        :param examples: A list of examples 
        :param batchsz: Batch size per step
        :param kwargs: See below
        
        :Keyword Arguments:
            * *shuffle* -- Shuffle the data per epoch? Defaults to `False`
            * *vec_alloc* -- Allocate a new tensor.  Defaults to ``numpy.zeros``
            * *vec_shape* -- Function to retrieve tensor shape.  Defaults to ``numpy.shape``
            * *trim* -- Trim batches to the maximum length seen in the batch (defaults to `False`)
                This can lead to batches being shorter than the maximum length provided to the system.
                Not supported in all frameworks.
            * *src_vec_trans* -- A transform function to use on the source tensor (`None`)
        """
        super(ExampleDataFeed, self).__init__()

        self.examples = examples
        self.batchsz = batchsz
        self.shuffle = bool(kwargs.get('shuffle', False))
        self.vec_alloc = kwargs.get('vec_alloc', np.zeros)
        self.vec_shape = kwargs.get('vec_shape', np.shape)
        self.src_vec_trans = kwargs.get('src_vec_trans', None)
        self.steps = int(math.floor(len(self.examples)/float(batchsz)))
        self.trim = bool(kwargs.get('trim', False))


class SeqLabelExamples(object):
    """Unstructured prediction examples
    
    Datasets of paired `(x, y)` data, where `x` is a tensor of data over time and `y` is a single label
    """
    SEQ = 0
    LABEL = 1

    def __init__(self, example_list, do_shuffle=True):
        self.example_list = example_list
        if do_shuffle:
            random.shuffle(self.example_list)

    def __getitem__(self, i):
        """Get a single example
        
        :param i: (``int``) simple index
        :return: an example
        """
        ex = self.example_list[i]
        return ex[SeqLabelExamples.SEQ], ex[SeqLabelExamples.LABEL]

    def __len__(self):
        """Number of examples
        
        :return: (``int``) length of data
        """
        return len(self.example_list)

    def width(self):
        """ Width of the temporal signal
        
        :return: (``int``) length
        """
        x, y = self.example_list[0]
        return len(x)

    def batch(self, start, batchsz, vec_alloc=np.empty):
        """Get a batch of data
        
        :param start: The step index
        :param batchsz: The batch size
        :param vec_alloc: A vector allocator, defaults to `numpy.empty`
        :return: batched x vector, batched y vector
        """
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
        """Function to produce a split of data based on a fraction
        
        :param data: Data to split
        :param splitfrac: (``float``) fraction of data to hold out
        :return: Two sets of label examples
        """
        numinst = len(data.examples)
        heldout = int(math.floor(numinst * (1-splitfrac)))
        heldout_ex = data.example_list[1:heldout]
        rest_ex = data.example_list[heldout:]
        return SeqLabelExamples(heldout_ex), SeqLabelExamples(rest_ex)


class SeqLabelDataFeed(ExampleDataFeed):
    """Data feed for :class:`SeqLabelExamples`
    """
    def __init__(self, examples, batchsz, **kwargs):
        super(SeqLabelDataFeed, self).__init__(examples, batchsz, **kwargs)

    def _batch(self, i):
        """
        Get a batch of data at step `i`
        :param i: (``int``) step index
        :return: A batch tensor x, batch tensor y
        """
        x, y = self.examples.batch(i, self.batchsz, self.vec_alloc)
        if self.src_vec_trans is not None:
            x = self.src_vec_trans(x)
        return {'x': x, 'y': y}


class SeqWordCharTagExamples(object):
    """Examples of sequences of words, characters and tags
    """
    SEQ_WORD = 0
    SEQ_CHAR = 1
    SEQ_TAG = 2
    SEQ_LEN = 3
    SEQ_ID = 4

    def __init__(self, example_list, do_shuffle=True, do_sort=True):
        """Constructor
        
        :param example_list:  A list of examples
        :param do_shuffle: (``bool``) Shuffle the data? Defaults to `True`
        :param do_sort: (``bool``) Sort the data.  Defaults to `True`
        """
        self.example_list = example_list
        if do_shuffle:
            random.shuffle(self.example_list)
        if do_sort:
            self.example_list = sorted(self.example_list, key=lambda x: x[SeqWordCharTagExamples.SEQ_LEN])

    def __getitem__(self, i):
        """Get `ith` example in order `SEQ_WORD`, `SEQ_CHAR`, `SEQ_TAG`, `SEQ_LEN`, `SEQ_ID`
        
        :param i: (``int``) index of example
        :return: example in order `SEQ_WORD`, `SEQ_CHAR`, `SEQ_TAG`, `SEQ_LEN`, `SEQ_ID`
        """
        ex = self.example_list[i]
        return ex[SeqWordCharTagExamples.SEQ_WORD], ex[SeqWordCharTagExamples.SEQ_CHAR], \
               ex[SeqWordCharTagExamples.SEQ_TAG], ex[SeqWordCharTagExamples.SEQ_LEN], \
               ex[SeqWordCharTagExamples.SEQ_ID]

    def __len__(self):
        """Get the number of examples
        
        :return: (``int``) number of examples
        """
        return len(self.example_list)

    def batch(self, start, batchsz, trim=False, vec_alloc=np.empty, vec_shape=np.shape):
        """Get a batch of data
        
        :param start: (``int``) The step index
        :param batchsz: (``int``) The batch size
        :param trim: (``bool``) Trim to maximum length in a batch
        :param vec_alloc: A vector allocator, defaults to `numpy.empty`
        :param vec_shape: A vector shape function, defaults to `numpy.shape`
        :return: batched `x` word vector, `x` character vector, batched `y` vector, `length` vector, `ids`
        """
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

        return {"x": xs, "xch": xs_ch, "y": ys, "lengths": length, "ids": ids}


    @staticmethod
    def valid_split(data, splitfrac=0.15):
        """Function to produce a split of data based on a fraction
        
        :param data: Data to split
        :param splitfrac: (``float``) fraction of data to hold out
        :return: Two sets of label examples
        """
        numinst = len(data.examples)
        heldout = int(math.floor(numinst * (1-splitfrac)))
        heldout_ex = data.example_list[1:heldout]
        rest_ex = data.example_list[heldout:]
        return SeqLabelExamples(heldout_ex), SeqLabelExamples(rest_ex)


class SeqWordCharLabelDataFeed(ExampleDataFeed):
    """Feed object for sequential prediction training data
    """
    def __init__(self, examples, batchsz, **kwargs):
        super(SeqWordCharLabelDataFeed, self).__init__(examples, batchsz, **kwargs)

    def _batch(self, i):
        return self.examples.batch(i, self.batchsz, self.trim, self.vec_alloc, self.vec_shape)


class Seq2SeqExamples(object):
    """Paired training examples
    """
    SRC = 0
    TGT = 1
    SRC_LEN = 2
    TGT_LEN = 3

    def __init__(self, example_list, do_shuffle=True, do_sort=True):
        """Constructor
        
        :param example_list: Training pair examples 
        :param do_shuffle: Shuffle the data (defaults to `True`)
        :param do_sort: Sort the data (defaults to `True`)
        """
        self.example_list = example_list
        if do_shuffle:
            random.shuffle(self.example_list)
        if do_sort:
            self.example_list = sorted(self.example_list, key=lambda x: x[Seq2SeqExamples.SRC_LEN])

    def __getitem__(self, i):
        """Get the `ith` example from the training data
        
        :param i: (``int``) integer offset
        :return: Example of `SRC`, `TGT`, `SRC_LEN`, `TGT_LEN`
        """
        ex = self.example_list[i]
        return ex[Seq2SeqExamples.SRC], ex[Seq2SeqExamples.TGT], ex[Seq2SeqExamples.SRC_LEN], ex[Seq2SeqExamples.TGT_LEN]

    def __len__(self):
        return len(self.example_list)

    def batch(self, start, batchsz, trim=False, vec_alloc=np.zeros):
        """Get a batch of data
        
        :param start: (``int``) The step index
        :param batchsz: (``int``) The batch size
        :param trim: (``bool``) Trim to maximum length in a batch
        :param vec_alloc: A vector allocator, defaults to `numpy.empty`
        :param vec_shape: A vector shape function, defaults to `numpy.shape`
        :return: batched source vector, target vector, source lengths, target lengths
        """
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
            srcs = srcs[:, 0:max_src_len]
            tgts = tgts[:, 0:max_tgt_len]

        return srcs, tgts, src_lens, tgt_lens


def reverse_2nd(vec):
    """Do time-reversal on numpy array of form `B x T`
    
    :param vec: vector to time-reverse
    :return: Time-reversed vector
    """
    return vec[:, ::-1]


class Seq2SeqDataFeed(ExampleDataFeed):
    """Data feed of paired examples
    """
    def __init__(self, examples, batchsz, **kwargs):
        super(Seq2SeqDataFeed, self).__init__(examples, batchsz, **kwargs)

    def _batch(self, i):
        src, tgt, src_len, tgt_len = self.examples.batch(i, self.batchsz, self.trim, self.vec_alloc)
        if self.src_vec_trans is not None:
            src = self.src_vec_trans(src)
        return {'src': src, 'dst': tgt, 'src_len': src_len, 'dst_len': tgt_len}


# This one is a little different at the moment
class SeqWordCharDataFeed(DataFeed):
    """Data feed to return language modeling training data
    """

    def __init__(self, x, xch, nbptt, batchsz, maxw):
        """Constructor
        
        :param x: word tensor
        :param xch: character tensor
        :param nbptt: Number of steps of BPTT
        :param batchsz: Batch size
        :param maxw: The maximum word length
        """
        super(SeqWordCharDataFeed, self).__init__()
        num_examples = x.shape[0]
        rest = num_examples // batchsz
        #if num_examples is divisible by batchsz * nbptt (equivalent to rest is divisible by nbptt), we have a problem. reduce rest in that case.
        if rest % nbptt == 0: 
            rest = rest-1

        self.steps = rest // nbptt
        #if num_examples is divisible by batchsz * nbptt (equivalent to rest is divisible by nbptt), we #have a problem. reduce rest in that case.
        if rest % nbptt == 0: 
            rest = rest-1

        self.stride_ch = nbptt * maxw

        trunc = rest*batchsz
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
        return {
            'x': self.x[:, i*self.nbptt:(i+1)*self.nbptt].reshape((self.batchsz, self.nbptt)),
            'xch': self.xch[:, i*self.stride_ch:(i+1)*self.stride_ch].reshape((self.batchsz, self.nbptt, self.wsz)),
            'y': self.x[:, i*self.nbptt+1:(i+1)*self.nbptt+1].reshape((self.batchsz, self.nbptt))
        }


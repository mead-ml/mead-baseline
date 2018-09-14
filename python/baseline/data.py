import random
import numpy as np
import math
from baseline.utils import export

__all__ = []
exporter = export(__all__)


@exporter
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


@exporter
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
        self.steps = int(math.floor(len(self.examples)/float(batchsz)))
        self.trim = bool(kwargs.get('trim', False))


@exporter
class DictExamples(object):
    """This object holds a list of dictionaries, and knows how to shuffle, sort and batch them
    """
    def __init__(self, example_list, do_shuffle=True, sort_key=None):
        """Constructor

        :param example_list:  A list of examples
        :param do_shuffle: (``bool``) Shuffle the data? Defaults to `True`
        :param do_sort: (``bool``) Sort the data.  Defaults to `True`
        """
        self.example_list = example_list

        if do_shuffle:
            random.shuffle(self.example_list)

        if sort_key is not None:
            self.example_list = sorted(self.example_list, key=lambda x: x[sort_key])

        self.sort_key = sort_key

    def __getitem__(self, i):
        """Get a single example
        
        :param i: (``int``) simple index
        :return: an example
        """
        return self.example_list[i]

    def __len__(self):
        """Number of examples
        
        :return: (``int``) length of data
        """
        return len(self.example_list)

    def _trim_batch(self, batch, keys, max_src_len):
        for k in keys:
            if len(batch[k].shape) == 3:
                batch[k] = batch[k][:, 0:max_src_len, :]
            elif len(batch[k].shape) == 2:
                batch[k] = batch[k][:, :max_src_len]
        return batch

    def batch(self, start, batchsz, trim=False):

        """Get a batch of data

        :param start: (``int``) The step index
        :param batchsz: (``int``) The batch size
        :param trim: (``bool``) Trim to maximum length in a batch
        :param vec_alloc: A vector allocator
        :param vec_shape: A vector shape function
        :return: batched `x` word vector, `x` character vector, batched `y` vector, `length` vector, `ids`
        """
        ex = self.example_list[start]
        keys = ex.keys()
        batch = {}

        for k in keys:
            batch[k] = []
        sz = len(self.example_list)
        idx = start * batchsz
        max_src_len = 0

        for i in range(batchsz):
            if idx >= sz:
                idx = 0

            ex = self.example_list[idx]
            for k in keys:
                batch[k] += [ex[k]]

            # Trim all batches along the sort_key if it exists
            if trim and self.sort_key is not None:
                max_src_len = max(max_src_len, ex[self.sort_key])
            idx += 1

        for k in keys:
            batch[k] = np.stack(batch[k])
        return self._trim_batch(batch, keys, max_src_len) if trim else batch


@exporter
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
        batch = self.examples.batch(i, self.batchsz, trim=self.trim)
        return batch


@exporter
class SeqWordCharLabelDataFeed(ExampleDataFeed):
    """Feed object for sequential prediction training data
    """
    def __init__(self, examples, batchsz, **kwargs):
        super(SeqWordCharLabelDataFeed, self).__init__(examples, batchsz, **kwargs)

    def _batch(self, i):
        return self.examples.batch(i, self.batchsz, self.trim, self.vec_alloc, self.vec_shape)


@exporter
class Seq2SeqExamples(object):
    """Paired training examples
    """
    def __init__(self, example_list, do_shuffle=True, sort_key=None):
        """Constructor
        
        :param example_list: Training pair examples 
        :param do_shuffle: Shuffle the data (defaults to `True`)
        :param do_sort: Sort the data (defaults to `True`)
        """
        self.example_list = example_list
        if do_shuffle:
            random.shuffle(self.example_list)
        if sort_key is not None:
            self.example_list = sorted(self.example_list, key=lambda x: x[sort_key])

    def __getitem__(self, i):
        """Get `ith` example

        :param i: (``int``) index of example
        :return: example dict
        """
        return self.example_list[i]

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


@exporter
def reverse_2nd(vec):
    """Do time-reversal on numpy array of form `B x T`
    
    :param vec: vector to time-reverse
    :return: Time-reversed vector
    """
    return vec[:, ::-1]


@exporter
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
@exporter
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
        self.steps = rest // nbptt
        #if num_examples is divisible by batchsz * nbptt (equivalent to rest is divisible by nbptt), we #have a problem. reduce rest in that case.
        if rest % nbptt == 0: 
            rest = rest-1

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
        return {
            'x': self.x[:, i*self.nbptt:(i+1)*self.nbptt].reshape((self.batchsz, self.nbptt)),
            'xch': self.xch[:, i*self.stride_ch:(i+1)*self.stride_ch].reshape((self.batchsz, self.nbptt, self.wsz)),
            'y': self.x[:, i*self.nbptt+1:(i+1)*self.nbptt+1].reshape((self.batchsz, self.nbptt))
        }


@exporter
class SeqCharDataFeed(DataFeed):
    """Data feed to return language modeling training data
    """

    def __init__(self, x, nbptt, batchsz):
        """Constructor

        :param x: word tensor
        :param xch: character tensor
        :param nbptt: Number of steps of BPTT
        :param batchsz: Batch size
        :param maxw: The maximum word length
        """
        super(SeqCharDataFeed, self).__init__()
        num_examples = x.shape[0]
        rest = num_examples // batchsz
        self.steps = rest // nbptt
        rest += 1
        trunc = batchsz * rest

        print('Truncating from %d to %d' % (num_examples, trunc))

        self.x = np.append(x, x[:batchsz])[:trunc].reshape((batchsz, rest))
        self.nbptt = nbptt
        self.batchsz = batchsz

    def _batch(self, i):

        return {

            'x': self.x[:, i*self.nbptt:(i+1)*self.nbptt].reshape((self.batchsz, self.nbptt)),
            'y': self.x[:, i*self.nbptt+1:(i+1)*self.nbptt+1].reshape((self.batchsz, self.nbptt))
        }

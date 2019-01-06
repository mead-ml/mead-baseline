import torch
import numpy as np
from baseline.utils import lookup_sentence, get_version, Offsets
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from eight_mile.pytorch.layers import *

PYT_MAJOR_VERSION = get_version(torch)


def sequence_mask(lengths, max_len=-1):
    lens = lengths.cpu()
    if max_len < 0:
        max_len = torch.max(lens).item()
    # 1 x T
    row = torch.arange(0, max_len).type_as(lens).view(1, -1)
    # B x 1
    col = lens.view(-1, 1)
    # Broadcast to B x T, compares increasing number to max
    mask = row < col
    return mask


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class SequenceCriterion(nn.Module):

    def __init__(self, LossFn=nn.NLLLoss, avg='token'):
        super(SequenceCriterion, self).__init__()
        if avg == 'token':
            # self.crit = LossFn(ignore_index=Offsets.PAD, reduction='elementwise-mean')
            self.crit = LossFn(ignore_index=Offsets.PAD, size_average=True)
            self._norm = self._no_norm
        else:
            self.crit = LossFn(ignore_index=Offsets.PAD, size_average=False)
            self._norm = self._batch_norm

    def _batch_norm(self, loss, inputs):
        return loss / inputs.size()[0]

    def _no_norm(self, loss, inputs):
        return loss

    def forward(self, inputs, targets):
        """Evaluate some loss over a sequence.

        :param inputs: torch.FloatTensor, [B, .., C] The scores from the model. Batch First
        :param targets: torch.LongTensor, The labels.

        :returns: torch.FloatTensor, The loss.
        """
        total_sz = targets.nelement()
        loss = self.crit(inputs.view(total_sz, -1), targets.view(total_sz))
        return self._norm(loss, inputs)


class StackedLSTMCell(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTMCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size=input_size, hidden_size=rnn_size, bias=False))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        hs, cs = [], []
        for i, layer in enumerate(self.layers):
            h_i, c_i = layer(input, (h_0[i], c_0[i]))
            input = h_i
            if i != self.num_layers - 1:
                input = self.dropout(input)
            hs.append(h_i)
            cs.append(c_i)

        hs = torch.stack(hs)
        cs = torch.stack(cs)

        return input, (hs, cs)


class StackedGRUCell(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRUCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size=input_size, hidden_size=rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0 = hidden
        hs = []
        for i, layer in enumerate(self.layers):
            h_i = layer(input, (h_0[i]))
            input = h_i
            if i != self.num_layers:
                input = self.dropout(input)
            hs.append(h_i)

        hs = torch.stack(hs)

        return input, hs


def pytorch_rnn_cell(insz, hsz, rnntype, nlayers, dropout):

    if rnntype == 'gru':
        rnn = StackedGRUCell(nlayers, insz, hsz, dropout)
    else:
        rnn = StackedLSTMCell(nlayers, insz, hsz, dropout)
    return rnn


def pytorch_embedding(weights, finetune=True):
    lut = nn.Embedding(weights.shape[0], weights.shape[1], padding_idx=0)
    del lut.weight
    lut.weight = nn.Parameter(torch.FloatTensor(weights),
                              requires_grad=finetune)
    return lut


def pytorch_conv1d(in_channels, out_channels, fsz, unif=0, padding=0, initializer=None):
    c = nn.Conv1d(in_channels, out_channels, fsz, padding=padding)
    if unif > 0:
        c.weight.data.uniform_(-unif, unif)
    elif initializer == "ortho":
        nn.init.orthogonal(c.weight)
    elif initializer == "he" or initializer == "kaiming":
        nn.init.kaiming_uniform(c.weight)
    else:
        nn.init.xavier_uniform_(c.weight)
    return c



def tie_weight(to_layer, from_layer):
    """Assigns a weight object to the layer weights.

    This method exists to duplicate baseline functionality across packages.

    :param to_layer: the pytorch layer to assign weights to  
    :param from_layer: pytorch layer to retrieve weights from  
    """
    to_layer.weight = from_layer.weight

def pytorch_clone_module(module_, N):
    return nn.ModuleList([copy.deepcopy(module_) for _ in range(N)])


def append2seq(seq, modules):

    for i, module in enumerate(modules):
        seq.add_module('%s-%d' % (str(module).replace('.', 'dot'), i), module)


def tensor_max(tensor):
    return tensor.max()


def tensor_shape(tensor):
    return tensor.size()


def tensor_reverse_2nd(tensor):
    idx = torch.LongTensor([i for i in range(tensor.size(1)-1, -1, -1)])
    return tensor.index_select(1, idx)


def long_0_tensor_alloc(dims, dtype=None):
    lt = long_tensor_alloc(dims)
    lt.zero_()
    return lt


def long_tensor_alloc(dims, dtype=None):
    if type(dims) == int or len(dims) == 1:
        return torch.LongTensor(dims)
    return torch.LongTensor(*dims)


# Some of this code is borrowed from here:
# https://github.com/rguthrie3/DeepLearningForNLPInPytorch
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.data[0]


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def vec_log_sum_exp(vec, dim):
    """Vectorized version of log-sum-exp

    :param vec: Vector
    :param dim: What dimension to operate on
    :return:
    """
    max_scores, idx = torch.max(vec, dim, keepdim=True)
    max_scores_broadcast = max_scores.expand_as(vec)
    return max_scores + torch.log(torch.sum(torch.exp(vec - max_scores_broadcast), dim, keepdim=True))

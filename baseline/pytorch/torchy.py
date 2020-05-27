import math
import copy
import os
import logging
import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from baseline.utils import lookup_sentence, get_version, Offsets
from eight_mile.pytorch.layers import *

PYT_MAJOR_VERSION = get_version(torch)

BaseLayer = nn.Module
TensorDef = torch.Tensor


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


def pytorch_rnn_cell(insz, hsz, rnntype, nlayers, dropout):

    if rnntype == 'gru':
        rnn = StackedGRUCell(nlayers, insz, hsz, dropout)
    else:
        rnn = StackedLSTMCell(nlayers, insz, hsz, dropout)
    return rnn


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

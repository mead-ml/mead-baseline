import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import *

class SequenceCriterion(nn.Module):

    def __init__(self, nc):
        super(SequenceCriterion, self).__init__()
        # Assume pad is zero element for now
        weight = torch.ones(nc)
        weight[0] = 0
        self.crit = nn.NLLLoss(weight, size_average=False)
    
    def forward(self, inputs, targets):
        # This is BxT, which is what we want!
        total_sz = targets.nelement()
        loss = self.crit(inputs.view(total_sz, -1), targets.view(total_sz))
        return loss


def _rnn(insz, hsz, rnntype, nlayers):

    if rnntype == 'gru':
        rnn = torch.nn.GRU(insz, hsz, nlayers)
    else:
        rnn = torch.nn.LSTM(insz, hsz, nlayers)
    return rnn

def _embedding(x2vec, finetune=True):
    dsz = x2vec.dsz
    lut = nn.Embedding(x2vec.vsz + 1, dsz, padding_idx=0)
    del lut.weight
    lut.weight = nn.Parameter(torch.FloatTensor(x2vec.weights),
                              requires_grad=finetune)
    return lut

def _append2seq(seq, modules):
    for module in modules:
        seq.add_module(str(module), module)

class Seq2SeqModel(nn.Module):

    def save(self, outdir, base):
        outname = '%s/%s.model' % (outdir, base)
        torch.save(self, outname)

    def create_loss(self):
        return SequenceCriterion(self.nc)

    @staticmethod
    def load(dirname, base):
        name = '%s/%s.model' % (dirname, base)
        return torch.load(name)

    # TODO: Add more dropout, BN
    def __init__(self, embed1, embed2, mxlen, hsz, nlayers, rnntype, batchfirst=True):
        super(Seq2SeqModel, self).__init__()
        dsz = embed1.dsz

        self.embed_in = _embedding(embed1)
        self.embed_out = _embedding(embed2)
        self.nc = embed2.vsz + 1            
        self.encoder_rnn = _rnn(dsz, hsz, rnntype, nlayers)
        self.decoder_rnn = _rnn(hsz, hsz, rnntype, nlayers)
        self.preds = nn.Linear(hsz, self.nc)
        self.batchfirst = batchfirst
        self.probs = nn.LogSoftmax()

    # Input better be xch, x
    def forward(self, input):
        rnn_enc_seq, final_encoder_state = self.encode(input[0])
        return self.decode(final_encoder_state, input[1])

    def decode(self, final_encoder_state, dst):
        if self.batchfirst is True:
            dst = dst.transpose(0, 1).contiguous()
        embed_out_seq = self.embed_out(dst)
        output, _ = self.decoder_rnn(embed_out_seq, final_encoder_state)

        # Reform batch as (T x B, D)
        pred = self.probs(self.preds(output.view(output.size(0)*output.size(1),
                                                 -1)))
        # back to T x B x H -> B x T x H
        pred = pred.view(output.size(0), output.size(1), -1)
        return pred.transpose(0, 1).contiguous() if self.batchfirst else pred

    def encode(self, src):
        if self.batchfirst is True:
            src = src.transpose(0, 1).contiguous()

        embed_in_seq = self.embed_in(src)
        return self.encoder_rnn(embed_in_seq)

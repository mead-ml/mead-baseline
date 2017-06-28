import torch
import numpy as np
from baseline.utils import lookup_sentence
import torch.autograd
import torch.nn as nn


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
            hs += [h_i]
            cs += [c_i]

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
            hs += [h_i]

        hs = torch.stack(hs)

        return input, hs

def _rnn_cell(insz, hsz, rnntype, nlayers, dropout):

    if rnntype == 'gru':
        rnn = StackedGRUCell(nlayers, insz, hsz, dropout)
    else:
        rnn = StackedLSTMCell(nlayers, insz, hsz, dropout)
    return rnn

def _embedding(x2vec, finetune=True):
    dsz = x2vec.dsz
    lut = nn.Embedding(x2vec.vsz + 1, dsz, padding_idx=0)
    del lut.weight
    lut.weight = nn.Parameter(torch.FloatTensor(x2vec.weights),
                              requires_grad=finetune)
    return lut


def _rnn(insz, hsz, rnntype, nlayers, dropout):

    if rnntype == 'gru':
        rnn = torch.nn.GRU(insz, hsz, nlayers, dropout=dropout)
    else:
        rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=dropout)
    return rnn

def append2seq(seq, modules):

    for i, module in enumerate(modules):
        seq.add_module('%s-%d' % (str(module), i), module)

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

# Mashed together from code using numpy only, hacked for th Tensors
def show_examples(use_gpu, model, es, rlut1, rlut2, embed2, mxlen, sample, prob_clip, max_examples, reverse):
    si = np.random.randint(0, len(es))

    src_array, tgt_array, src_len, _ = es[si]

    if max_examples > 0:
        max_examples = min(max_examples, src_array.size(0))
        src_array = src_array[0:max_examples]
        tgt_array = tgt_array[0:max_examples]
        src_len = src_len[0:max_examples]

    GO = embed2.vocab['<GO>']
    EOS = embed2.vocab['<EOS>']

    if use_gpu:
        src_array = src_array.cuda()
    
    for src_len,src_i,tgt_i in zip(src_len, src_array, tgt_array):

        print('========================================================================')

        sent = lookup_sentence(rlut1, src_i.cpu().numpy(), reverse=reverse)
        print('[OP] %s' % sent)
        sent = lookup_sentence(rlut2, tgt_i)
        print('[Actual] %s' % sent)
        dst_i = torch.zeros(1, mxlen).long()
        if use_gpu:
            dst_i = dst_i.cuda()

        next_value = GO
        src_i = src_i.view(1, -1)
        for j in range(mxlen):
            dst_i[0,j] = next_value
            probv = model((torch.autograd.Variable(src_i), torch.autograd.Variable(dst_i)))
            output = probv.squeeze()[j]
            if sample is False:
                _, next_value = torch.max(output, 0)
                next_value = int(next_value.data[0])
            else:
                probs = output.data.exp()
                # This is going to zero out low prob. events so they are not
                # sampled from
                best, ids = probs.topk(prob_clip, 0, largest=True, sorted=True)
                probs.zero_()
                probs.index_copy_(0, ids, best)
                probs.div_(torch.sum(probs))
                fv = torch.multinomial(probs, 1)[0]
                next_value = fv

            if next_value == EOS:
                break

        sent = lookup_sentence(rlut2, dst_i.squeeze())
        print('Guess: %s' % sent)
        print('------------------------------------------------------------------------')

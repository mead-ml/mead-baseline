import torch
import numpy as np
from baseline.utils import lookup_sentence, get_version, revlut
from torch.autograd import Variable
import torch.autograd
import torch.nn as nn


PYT_MAJOR_VERSION = get_version(torch)


def sequence_mask(lengths):
    max_len = torch.max(lengths)
    # 1 x T
    row = Variable(torch.arange(0, max_len.data[0]), requires_grad=False).view(1, -1)
    # B x 1
    col = lengths.view(-1, 1).float()
    # Broadcast to B x T, compares increasing number to max
    mask = row < col
    return mask.float()


def attention_mask(scores, mask, value=100000):
    # Make padded scores really low so they become 0 in softmax
    scores = scores * mask + (mask - 1) * value
    return scores


def classify_bt(model, batch_time):
    tensor = torch.from_numpy(batch_time) if type(batch_time) == np.ndarray else batch_time
    probs = model(torch.autograd.Variable(tensor, requires_grad=False).cuda()).exp().data
    probs.div_(torch.sum(probs))
    results = []
    batchsz = probs.size(0)
    for b in range(batchsz):
        outcomes = [(model.labels[id_i], prob_i) for id_i, prob_i in enumerate(probs[b])]
        results.append(outcomes)
    return results


def predict_seq_bt(model, x, xch, lengths):
    x_t = torch.from_numpy(x) if type(x) == np.ndarray else x
    xch_t = torch.from_numpy(xch) if type(xch) == np.ndarray else xch
    len_v = torch.from_numpy(lengths) if type(lengths) == np.ndarray else lengths
    x_v = torch.autograd.Variable(x_t, requires_grad=False).cuda()
    xch_v = torch.autograd.Variable(xch_t, requires_grad=False).cuda()
    #len_v = torch.autograd.Variable(len_t, requires_grad=False)
    results = model((x_v, xch_v, len_v))
    #print(results)
    #if type(x) == np.ndarray:
    #    # results = results.cpu().numpy()
    #    # Fix this to not be greedy
    #    results = np.argmax(results, -1)

    return results


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

    def __init__(self, LossFn=nn.NLLLoss):
        super(SequenceCriterion, self).__init__()
        self.crit = LossFn(ignore_index=0, size_average=False)

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


def pytorch_rnn_cell(insz, hsz, rnntype, nlayers, dropout):

    if rnntype == 'gru':
        rnn = StackedGRUCell(nlayers, insz, hsz, dropout)
    else:
        rnn = StackedLSTMCell(nlayers, insz, hsz, dropout)
    return rnn


def pytorch_embedding(x2vec, finetune=True):
    dsz = x2vec.dsz
    lut = nn.Embedding(x2vec.vsz + 1, dsz, padding_idx=0)
    del lut.weight
    lut.weight = nn.Parameter(torch.FloatTensor(x2vec.weights),
                              requires_grad=finetune)
    return lut


def pytorch_activation(name="relu"):
    if name == "tanh":
        return nn.Tanh()
    if name == "hardtanh":
        return nn.Hardtanh()
    if name == "prelu":
        return nn.PReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "log_sigmoid":
        return nn.LogSigmoid()
    return nn.ReLU()


def pytorch_conv1d(in_channels, out_channels, fsz, unif=0, padding=0, initializer=None):
    c = nn.Conv1d(in_channels, out_channels, fsz, padding=padding)
    if unif > 0:
        c.weight.data.uniform_(-unif, unif)
    elif initializer == "ortho":
        nn.init.orthogonal(c.weight)
    elif initializer == "he" or initializer == "kaiming":
        nn.init.kaiming_uniform(c.weight)
    else:
        nn.init.xavier_uniform(c.weight)
    return c


def pytorch_linear(in_sz, out_sz, unif=0, initializer=None):
    l = nn.Linear(in_sz, out_sz)
    if unif > 0:
        l.weight.data.uniform_(-unif, unif)
    elif initializer == "ortho":
        nn.init.orthogonal(l.weight)
    elif initializer == "he" or initializer == "kaiming":
        nn.init.kaiming_uniform(l.weight)
    else:
        nn.init.xavier_uniform(l.weight)

    l.bias.data.zero_()
    return l


def _cat_dir(h):
    return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], dim=-1)


class BiRNNWrapper(nn.Module):

    def __init__(self, rnn, nlayers):
        super(BiRNNWrapper, self).__init__()
        self.rnn = rnn
        self.nlayers = nlayers

    def forward(self, seq):
        output, hidden = self.rnn(seq)
        if isinstance(hidden, tuple):
            hidden = tuple(_cat_dir(h) for h in hidden)
        else:
            hidden = _cat_dir(hidden)
        return output, hidden


def pytorch_rnn(insz, hsz, rnntype, nlayers, dropout):

    if rnntype == 'gru':
        rnn = torch.nn.GRU(insz, hsz, nlayers, dropout=dropout)
    elif rnntype == 'blstm':
        rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=dropout, bidirectional=True)
        rnn = BiRNNWrapper(rnn, nlayers)
    else:
        rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=dropout)
    return rnn


class Highway(nn.Module):

    def __init__(self,
                 input_size):
        super(Highway, self).__init__()
        self.proj = nn.Linear(input_size, input_size)
        self.transform = nn.Linear(input_size, input_size)
        self.transform.bias.data.fill_(-2.0)

    def forward(self, input):
        proj_result = nn.functional.relu(self.proj(input))
        proj_gate = nn.functional.sigmoid(self.transform(input))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * input)
        return gated


def pytorch_lstm(insz, hsz, rnntype, nlayers, dropout, unif=0, batch_first=False, initializer=None):
    ndir = 2 if rnntype.startswith('b') else 1
    #print('ndir: %d, rnntype: %s, nlayers: %d, dropout: %.2f, unif: %.2f' % (ndir, rnntype, nlayers, dropout, unif))
    rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=dropout, bidirectional=True if ndir > 1 else False, batch_first=batch_first)#, bias=False)
    if unif > 0:
        for weight in rnn.parameters():
            weight.data.uniform_(-unif, unif)
    elif initializer == "ortho":
        nn.init.orthogonal(rnn.weight_hh_l0)
        nn.init.orthogonal(rnn.weight_ih_l0)
    elif initializer == "he" or initializer == "kaiming":
        nn.init.kaiming_uniform(rnn.weight_hh_l0)
        nn.init.kaiming_uniform(rnn.weight_ih_l0)
    else:
        nn.init.xavier_uniform(rnn.weight_hh_l0)
        nn.init.xavier_uniform(rnn.weight_ih_l0)

    return rnn, ndir*hsz


def pytorch_prepare_optimizer(model, **kwargs):

    mom = kwargs.get('mom', 0.9)
    optim = kwargs.get('optim', 'sgd')
    eta = kwargs.get('eta', kwargs.get('lr', 0.01))
    decay_rate = float(kwargs.get('decay_rate', 0.0))
    decay_type = kwargs.get('decay_type', None)

    if optim == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=eta)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=eta)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=eta)
    elif optim == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters(), lr=eta)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=mom)

    scheduler = None
    if decay_rate > 0.0 and decay_type is not None:
        if decay_type == 'invtime':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=decay_rate)

    return optimizer, scheduler


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


def prepare_src(model, tokens, mxlen=100):
    src_vocab = model.get_src_vocab()
    length = min(len(tokens), mxlen)
    x = torch.LongTensor(length).zero_()

    for j in range(length):
        word = tokens[j]
        if word not in src_vocab:
            if word != '':
                print(word)
                idx = 0
        else:
            idx = src_vocab[word]
        x[j] = idx
    return torch.autograd.Variable(x.view(-1, 1))


def beam_decode(model, src, K):
    with torch.no_grad():
        T = src.size(1)
        # Transpose in here?
        context, h_i = model.encode(src)
        dst_vocab = model.get_dst_vocab()
        GO = dst_vocab['<GO>']
        EOS = dst_vocab['<EOS>']

        paths = [[GO] for _ in range(K)]
        # K
        scores = torch.FloatTensor([0. for _ in range(K)])
        if src.is_cuda:
            scores = scores.cuda()
        # TBH
        context = torch.autograd.Variable(context.data.repeat(1, K, 1))
        h_i = (torch.autograd.Variable(h_i[0].data.repeat(1, K, 1)), torch.autograd.Variable(h_i[1].data.repeat(1, K, 1)))
        h_i, dec_out = model.bridge(h_i, context)

        for i in range(T):
            lst = [path[-1] for path in paths]
            dst = torch.LongTensor(lst).type(src.type())
            mask_eos = dst == EOS
            mask_pad = dst == 0
            dst = dst.view(1, K)
            var = torch.autograd.Variable(dst)
            dec_out, h_i = model.decode_rnn(context, h_i, dec_out, var)
            # 1 x K x V
            wll = model.prediction(dec_out).data
            # Just mask wll against end data
            V = wll.size(-1)
            dec_out = dec_out.squeeze(0)  # get rid of T=t dimension
            # K x V
            wll = wll.squeeze(0)  # get rid of T=t dimension

            if i > 0:
                expanded_history = scores.unsqueeze(1).expand_as(wll)
                wll.masked_fill_(mask_eos | mask_pad, 0)
                sll = wll + expanded_history
            else:
                sll = wll[0]

            flat_sll = sll.view(-1)
            best, best_idx = flat_sll.squeeze().topk(K, 0)
            best_beams = best_idx / V
            best_idx = best_idx % V
            new_paths = []
            for j, beam_id in enumerate(best_beams):
                new_paths.append(paths[beam_id] + [best_idx[j]])
                scores[j] = best[j]

            # Copy the beam state of the winners
            for hc in h_i:  # iterate over h, c
                old_beam_state = hc.clone()
                for i, beam_id in enumerate(best_beams):
                    H = hc.size(2)
                    src_beam = old_beam_state.view(-1, K, H)[:, beam_id]
                    dst_beam = hc.view(-1, K, H)[:, i]
                    dst_beam.data.copy_(src_beam.data)
            paths = new_paths

        return paths, scores


def beam_decode_tokens(model, src_tokens, K, idx2word, mxlen=50):
    src = prepare_src(model, src_tokens, mxlen)
    paths, scores = beam_decode(model, src, K)
    path_str = []
    for j, path in enumerate(paths):
        path_str.append([idx2word[i] for i in path])
    return path_str, scores
    #return beam_decode(model, src, K)

# Mashed together from code using numpy only, hacked for th Tensors
# This function should never be used for decoding.  It exists only so that the training model can greedily decode
# It is super slow and doesnt use maintain a beam of hypotheses
def show_examples_pytorch(model, es, rlut1, rlut2, embed2, mxlen, sample, prob_clip, max_examples, reverse):
    si = np.random.randint(0, len(es))

    batch_dict = es[si]

    src_array = batch_dict['src']
    tgt_array = batch_dict['dst']
    src_len = batch_dict['src_len']

    if max_examples > 0:
        max_examples = min(max_examples, src_array.size(0))
        src_array = src_array[0:max_examples]
        tgt_array = tgt_array[0:max_examples]
        src_len = src_len[0:max_examples]

    # TODO: fix this, check for GPU first
    src_array = src_array.cuda()
    
    for src_len, src_i, tgt_i in zip(src_len, src_array, tgt_array):

        print('========================================================================')

        sent = lookup_sentence(rlut1, src_i.cpu().numpy(), reverse=reverse)
        print('[OP] %s' % sent)
        sent = lookup_sentence(rlut2, tgt_i)
        print('[Actual] %s' % sent)

        dst_i, scores = beam_decode(model, torch.autograd.Variable(src_i.view(1, -1), requires_grad=False), 1)
        sent = lookup_sentence(rlut2, dst_i[0])
        print('Guess: %s' % sent)
        print('------------------------------------------------------------------------')

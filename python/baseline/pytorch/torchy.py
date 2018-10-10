import torch
import numpy as np
from baseline.utils import lookup_sentence, get_version
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

PYT_MAJOR_VERSION = get_version(torch)


def sequence_mask(lengths):
    lens = lengths.cpu()
    max_len = torch.max(lens)
    # 1 x T
    row = torch.arange(0, max_len.item()).type_as(lens).view(1, -1)
    # B x 1
    col = lens.view(-1, 1)
    # Broadcast to B x T, compares increasing number to max
    mask = row < col
    return mask


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


class VariationalDropout(nn.Module):
    """Inverted dropout that applies the same mask at each time step."""

    def __init__(self, p=0.5):
        """Variational Dropout

        :param p: float, the percentage to drop
        """
        super(VariationalDropout, self).__init__()
        self.p = p

    def extra_repr(self):
        return 'p=%.1f' % self.p

    def forward(self, input):
        if not self.training:
            return input
        # Create a mask that covers a single time step
        mask = torch.zeros(1, input.size(1), input.size(2)).bernoulli_(1 - self.p).to(input.device)
        mask = mask / self.p
        # Broadcast the mask over the sequence
        return mask * input


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
    lut = nn.Embedding(x2vec.vsz, x2vec.dsz, padding_idx=0)
    del lut.weight
    lut.weight = nn.Parameter(torch.FloatTensor(x2vec.weights),
                              requires_grad=finetune)
    return lut


def pytorch_activation(name="relu"):
    if name == "tanh":
        return nn.Tanh()
    if name == "identity":
        return nn.Identity()
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
        nn.init.xavier_uniform_(c.weight)
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
        nn.init.xavier_uniform_(l.weight)

    l.bias.data.zero_()
    return l


def pytorch_clone_module(module_, N):
    return nn.ModuleList([copy.deepcopy(module_) for _ in range(N)])


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
    if nlayers == 1:
        dropout = 0.0

    if rnntype == 'gru':
        rnn = torch.nn.GRU(insz, hsz, nlayers, dropout=dropout)
    elif rnntype == 'blstm':
        rnn = torch.nn.LSTM(insz, hsz//2, nlayers, dropout=dropout, bidirectional=True)
        rnn = BiRNNWrapper(rnn, nlayers)
    else:
        rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=dropout)
    return rnn


class ParallelConv(nn.Module):

    def __init__(self, insz, outsz, filtsz, activation_type, pdrop):
        super(ParallelConv, self).__init__()
        convs = []
        outsz_filts = outsz

        if type(outsz) == int:
            outsz_filts = len(filtsz) * [outsz]

        self.outsz = sum(outsz_filts)
        for i, fsz in enumerate(filtsz):
            pad = fsz//2
            conv = nn.Sequential(
                nn.Conv1d(insz, outsz_filts[i], fsz, padding=pad),
                pytorch_activation(activation_type)
            )
            convs.append(conv)
            # Add the module so its managed correctly
        self.convs = nn.ModuleList(convs)
        self.conv_drop = nn.Dropout(pdrop)

    def forward(self, input_bct):
        mots = []
        for conv in self.convs:
            # In Conv1d, data BxCxT, max over time
            conv_out = conv(input_bct)
            mot, _ = conv_out.max(2)
            mots.append(mot)
        mots = torch.cat(mots, 1)
        return self.conv_drop(mots)


class Highway(nn.Module):

    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.proj = nn.Linear(input_size, input_size)
        self.transform = nn.Linear(input_size, input_size)
        self.transform.bias.data.fill_(-2.0)

    def forward(self, input):
        proj_result = nn.functional.relu(self.proj(input))
        proj_gate = nn.functional.sigmoid(self.transform(input))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * input)
        return gated


class SkipConnection(nn.Module):

    def __init__(self, input_size, activation='relu'):
        super(SkipConnection, self).__init__()
        self.proj = nn.Linear(input_size, input_size)
        self.activation_fn = pytorch_activation(activation)

    def forward(self, input):
        return self.activation_fn(self.proj(input)) + input


class LayerNorm(nn.Module):
    """
    Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization`_ .

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x]} + \epsilon} * \gamma + \beta

    This is provided in pytorch's master, and can be replaced in the near future.
    For the time, being, this code is adapted from:
    http://nlp.seas.harvard.edu/2018/04/03/attention.html
    https://github.com/pytorch/pytorch/pull/2019
    """
    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(num_features))
        self.b = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = ((x - mean).pow(2).sum(-1, keepdim=True).div(x.size(-1) - 1) + self.eps).sqrt()
        d = (std + self.eps) + self.b
        return self.a * (x - mean) / d


def pytorch_lstm(insz, hsz, rnntype, nlayers, dropout, unif=0, batch_first=False, initializer=None):
    if nlayers == 1:
        dropout = 0.0
    ndir = 2 if rnntype.startswith('b') else 1
    layer_hsz = hsz // ndir
    rnn = torch.nn.LSTM(insz, layer_hsz, nlayers, dropout=dropout, bidirectional=True if ndir > 1 else False, batch_first=batch_first)#, bias=False)
    if initializer == "ortho":
        nn.init.orthogonal(rnn.weight_hh_l0)
        nn.init.orthogonal(rnn.weight_ih_l0)
    elif initializer == "he" or initializer == "kaiming":
        nn.init.kaiming_uniform(rnn.weight_hh_l0)
        nn.init.kaiming_uniform(rnn.weight_ih_l0)
    elif unif > 0:
        for weight in rnn.parameters():
            weight.data.uniform_(-unif, unif)
    else:
        nn.init.xavier_uniform_(rnn.weight_hh_l0)
        nn.init.xavier_uniform_(rnn.weight_ih_l0)

    return rnn


class LSTMEncoder(nn.Module):

    def __init__(self, insz, hsz, rnntype, nlayers, dropout, residual=False, unif=0, initializer=None):
        super(LSTMEncoder, self).__init__()
        self.residual = residual
        self.rnn = pytorch_lstm(insz, hsz, rnntype, nlayers, dropout, unif, False, initializer)

    def forward(self, tbc, lengths):

        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths.tolist())
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        return output + tbc if self.residual else output


class BaseAttention(nn.Module):

    def __init__(self, hsz):
        super(BaseAttention, self).__init__()
        self.hsz = hsz
        self.W_c = nn.Linear(2 * self.hsz, hsz, bias=False)

    def forward(self, query_t, keys_bth, values_bth, keys_mask=None):
        # Output(t) = B x H x 1
        # Keys = B x T x H
        # a = B x T x 1
        a = self._attention(query_t, keys_bth, keys_mask)
        attended = self._update(a, query_t, values_bth)

        return attended

    def _attention(self, query_t, keys_bth, keys_mask):
        pass

    def _update(self, a, query_t, values_bth):
        # a = B x T
        # Want to apply over context, scaled by a
        # (B x 1 x T) (B x T x H) = (B x 1 x H)
        a = a.view(a.size(0), 1, a.size(1))
        c_t = torch.bmm(a, values_bth).squeeze(1)

        attended = torch.cat([c_t, query_t], -1)
        attended = F.tanh(self.W_c(attended))
        return attended


class LuongDotProductAttention(BaseAttention):

    def __init__(self, hsz):
        super(LuongDotProductAttention, self).__init__(hsz)

    def _attention(self, query_t, keys_bth, keys_mask):
        a = torch.bmm(keys_bth, query_t.unsqueeze(2))
        a = a.squeeze(2).masked_fill(keys_mask == 0, -1e9)
        a = F.softmax(a, dim=-1)
        return a


class ScaledDotProductAttention(BaseAttention):

    def __init__(self, hsz):
        super(ScaledDotProductAttention, self).__init__(hsz)

    def _attention(self, query_t, keys_bth, keys_mask):
        a = torch.bmm(keys_bth, query_t.unsqueeze(2)) / math.sqrt(self.hsz)
        a = a.squeeze(2).masked_fill(keys_mask == 0, -1e9)
        a = F.softmax(a, dim=-1)
        return a


class LuongGeneralAttention(BaseAttention):

    def __init__(self, hsz):
        super(LuongGeneralAttention, self).__init__(hsz)
        self.W_a = nn.Linear(self.hsz, self.hsz, bias=False)

    def _attention(self, query_t, keys_bth, keys_mask):
        a = torch.bmm(keys_bth, self.W_a(query_t).unsqueeze(2))
        a = a.squeeze(2).masked_fill(keys_mask == 0, -1e9)
        a = F.softmax(a, dim=-1)
        return a


class BahdanauAttention(BaseAttention):

    def __init__(self, hsz):
        super(BahdanauAttention, self).__init__(hsz)
        self.hsz = hsz
        self.W_a = nn.Linear(self.hsz, self.hsz, bias=False)
        self.E_a = nn.Linear(self.hsz, self.hsz, bias=False)
        self.v = nn.Linear(self.hsz, 1, bias=False)

    def _attention(self, query_t, keys_bth, keys_mask):
        B, T, H = keys_bth.shape
        q = self.W_a(query_t.view(-1, self.hsz)).view(B, 1, H)
        u = self.E_a(keys_bth.contiguous().view(-1, self.hsz)).view(B, T, H)
        z = F.tanh(q + u)
        a = self.v(z.view(-1, self.hsz)).view(B, T)
        a.masked_fill(keys_mask == 0, -1e9)
        a = F.softmax(a, dim=-1)
        return a

    def _update(self, a, query_t, values_bth):
        # a = B x T
        # Want to apply over context, scaled by a
        # (B x 1 x T) (B x T x H) = (B x 1 x H) -> (B x H)
        a = a.view(a.size(0), 1, a.size(1))
        c_t = torch.bmm(a, values_bth).squeeze(1)
        # (B x 2H)
        attended = torch.cat([c_t, query_t], -1)
        attended = self.W_c(attended)
        return attended


class NoamOpt(object):

    """Introduced in the Transformer paper, increase learning rate linearly during warmup, then decrease

    The optimizer wraps Adam, and increases the learning rate linearly during the warmup period,
    and then decreases it proportional to the sqrt of the step

    """
    def __init__(self, d_model, params, warmup_steps=4000):
        self.optimizer = torch.optim.Adam(params, lr=0, betas=(0.9, 0.98), eps=1e-9)
        self.step_num = 0
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.current_lr = 0

    def step(self):
        """Runs at every step and updates the learning rate

        :return:
        """
        self.step_num += 1
        lr = self.d_model**(-0.5) * min(self.step_num**(-0.5), self.step_num*self.warmup_steps**(-.5))
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.current_lr = lr
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


def pytorch_prepare_optimizer(model, **kwargs):

    weight_decay = kwargs.get('weight_decay', 0)
    mom = kwargs.get('mom', 0.9)
    optim = kwargs.get('optim', 'sgd')
    eta = kwargs.get('eta', kwargs.get('lr', 0.01))
    decay_rate = float(kwargs.get('decay_rate', 0.0))
    decay_type = kwargs.get('decay_type', None)

    if optim == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=eta, weight_decay=weight_decay)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=eta, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=eta, weight_decay=weight_decay)
    elif optim == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters(), lr=eta, weight_decay=weight_decay)
    elif optim == 'noam':
        print('Using NoamOpt, lr will be ignored')
        d_model = kwargs['d_model']
        warmup_steps = kwargs.get('warmup_steps', 4000)
        optimizer = NoamOpt(d_model, model.parameters(), warmup_steps)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=mom, weight_decay=weight_decay)

    scheduler = None
    if decay_rate > 0.0 and decay_type is not None:
        if decay_type == 'invtime':
            gamma = 1.0 / (1.0 + 1.0 * decay_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=gamma)

    return optimizer, scheduler


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


def show_examples_pytorch(model, es, rlut1, rlut2, vocab, mxlen, sample, prob_clip, max_examples, reverse):
    si = np.random.randint(0, len(es))

    batch_dict = es[si]

    src_len_key = model.src_lengths_key
    src_field = src_len_key.split('_')[0]

    if max_examples > 0:
        max_examples = min(max_examples, batch_dict[src_field].shape[0])

    for i in range(max_examples):

        example = {}
        # Batch first, so this gets a single example at once
        for k, value in batch_dict.items():
            v = value[i]
            example[k] = v.reshape((1,) + v.shape)

        print('========================================================================')
        sent = lookup_sentence(rlut1, example[src_field].squeeze(), reverse=reverse)
        print('[OP] %s' % sent)
        sent = lookup_sentence(rlut2, example['tgt'].squeeze())
        print('[Actual] %s' % sent)

        dst_i = model.run(example)[0][0]
        dst_i = [idx.item() for idx in dst_i]
        sent = lookup_sentence(rlut2, dst_i)
        print('Guess: %s' % sent)
        print('------------------------------------------------------------------------')


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


class EmbeddingsContainer(nn.Module):
    def __init__(self):
        super(EmbeddingsContainer, self).__init__()

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self.add_module(key, module)

    def __delitem__(self, key):
        del self._modules[key]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __contains__(self, key):
        return key in self._modules

    def clear(self):
        self._modules.clear()

    def pop(self, key):
        v = self[key]
        del self[key]
        return v

    def keys(self):
        return self._modules.keys()

    def items(self):
        return self._modules.items()

    def values(self):
        return self._modules.values()

    def update(self, modules):
        raise Exception('Not implemented')

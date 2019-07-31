import math
import copy
import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from baseline.utils import lookup_sentence, get_version, Offsets


PYT_MAJOR_VERSION = get_version(torch)


def sequence_mask(lengths, max_len=-1):
    """Produce a sequence mask BxT

    :param lengths: The lengths for each temporal vector
    :param max_len: The maximum length.  If less than 0, determine from batch max vec len
    :return: A mask of 1s if valid, 0s o.w.
    """
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


def tie_weight(to_layer, from_layer):
    """Assigns a weight object to the layer weights.

    This method exists to duplicate baseline functionality across packages.

    :param to_layer: the pytorch layer to assign weights to  
    :param from_layer: pytorch layer to retrieve weights from  
    """
    to_layer.weight = from_layer.weight


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
    elif rnntype == 'bgru':
        rnn = torch.nn.GRU(insz, hsz//2, nlayers, dropout=dropout, bidirectional=True)
        rnn = BiRNNWrapper(rnn, nlayers)
    else:
        rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=dropout)
    return rnn


class ConvEncoder(nn.Module):
    def __init__(self, insz, outsz, filtsz, pdrop, activation_type='relu'):
        super(ConvEncoder, self).__init__()
        self.outsz = outsz
        pad = filtsz//2
        self.conv = nn.Conv1d(insz, outsz, filtsz, padding=pad)
        self.act = pytorch_activation(activation_type)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, input_bct):
        conv_out = self.act(self.conv(input_bct))
        return self.dropout(conv_out)


class ConvEncoderStack(nn.Module):

    def __init__(self, insz, outsz, filtsz, pdrop, layers=1, activation_type='relu'):
        super(ConvEncoderStack, self).__init__()

        first_layer = ConvEncoder(insz, outsz, filtsz, pdrop, activation_type)
        subsequent_layer = ResidualBlock(ConvEncoder(outsz, outsz, filtsz, pdrop, activation_type))
        self.layers = nn.ModuleList([first_layer] + [copy.deepcopy(subsequent_layer) for _ in range(layers-1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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


class ResidualBlock(nn.Module):

    def __init__(self, layer):
        super(ResidualBlock, self).__init__()
        self.layer = layer

    def forward(self, input):
        return input + self.layer(input)


class SkipConnection(ResidualBlock):

    def __init__(self, input_size, activation='relu'):
        super(SkipConnection, self).__init__(None)
        self.layer = nn.Sequential(
            nn.Linear(input_size, input_size),
            pytorch_activation(activation)
        )


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

        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        return output + tbc if self.residual else output


class BaseAttention(nn.Module):

    def __init__(self, hsz):
        """Construct a base attention model

        :param hsz: (`int`) The number of hidden units
        """
        super(BaseAttention, self).__init__()
        self.hsz = hsz
        self.W_c = nn.Linear(2 * self.hsz, hsz, bias=False)

    def forward(self, query_t, keys_bth, values_bth, keys_mask=None):
        """Take a query at time t, keys and values and produce an output

        :param query_t: (BxH) The query at time t
        :param keys_bth: (BxTxH) The keys
        :param values_bth: (BxTxH) The values
        :param keys_mask: (BxT) Mask of 1s where input is valid, 0 o.w.
        :return: Outputs of attention at the current timestep
        """
        # Output(t) = B x H x 1
        # Keys = B x T x H
        # a = B x T
        a = self._attention(query_t, keys_bth, keys_mask)
        attended = self._update(a, query_t, values_bth)
        return attended

    def _attention(self, query_t, keys_bth, keys_mask):
        """Perform attention between query and keys

        :param query_t: (BxH) The query at time t
        :param keys_bth: (BxTxH) The keys
        :param keys_mask:  (BxT) The mask
        :return: (BxT) attended output at time t
        """
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
        # First, unsqueeze so we have BxHx1
        # A BMM yields BxTx1
        a = torch.bmm(keys_bth, query_t.unsqueeze(2))
        # Now squeeze A down to BxT and apply the mask
        a = a.squeeze(2).masked_fill(keys_mask == 0, -1e9)
        # Do a softmax over time
        a = F.softmax(a, dim=-1)
        return a


class ScaledDotProductAttention(BaseAttention):

    def __init__(self, hsz):
        super(ScaledDotProductAttention, self).__init__(hsz)

    def _attention(self, query_t, keys_bth, keys_mask):
        # This is almost identical to Luong but we scale after the BMM
        a = torch.bmm(keys_bth, query_t.unsqueeze(2)) / math.sqrt(self.hsz)
        a = a.squeeze(2).masked_fill(keys_mask == 0, -1e9)
        a = F.softmax(a, dim=-1)
        return a


class LuongGeneralAttention(BaseAttention):

    def __init__(self, hsz):
        super(LuongGeneralAttention, self).__init__(hsz)
        self.W_a = nn.Linear(self.hsz, self.hsz, bias=False)

    def _attention(self, query_t, keys_bth, keys_mask):
        # This is almost like LuongDotProductAttention but we have an additional projection to apply to the input
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
        # Start with a projection, and get a Bx1xH output of query vector
        q = self.W_a(query_t.view(-1, self.hsz)).view(B, 1, H)
        # Project the keys as well, and view as BxTxH
        u = self.E_a(keys_bth.contiguous().view(-1, self.hsz)).view(B, T, H)
        # Add together, with a broadcast and get apply non-linearity
        z = torch.tanh(q + u)
        # Reshape as (BxT)xH and apply linear transformation to get back weighting
        a = self.v(z.view(-1, self.hsz)).view(B, T)
        a = a.masked_fill(keys_mask == 0, -1e9)
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


def unsort_batch(batch, perm_idx):
    """Undo the sort on a batch of tensors done for packing the data in the RNN.

    :param batch: `torch.Tensor`: The batch of data batch first `[B, ...]`
    :param perm_idx: `torch.Tensor`: The permutation index returned from the torch.sort.

    :returns: `torch.Tensor`: The batch in the original order.
    """
    # Add ones to the shape of the perm_idx until it can broadcast to the batch
    perm_idx = perm_idx.to(batch.device)
    diff = len(batch.shape) - len(perm_idx.shape)
    extra_dims = [1] * diff
    perm_idx = perm_idx.view([-1] + extra_dims)
    return batch.scatter_(0, perm_idx.expand_as(batch), batch)

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from baseline.utils import listify, Offsets
from baseline.utils import transition_mask as transition_mask_np
import torch.autograd


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


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# Some of this code is borrowed from here:
# https://github.com/rguthrie3/DeepLearningForNLPInPytorch
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.data[0]


def vec_log_sum_exp(vec, dim):
    """Vectorized version of log-sum-exp

    :param vec: Vector
    :param dim: What dimension to operate on
    :return:
    """
    max_scores, idx = torch.max(vec, dim, keepdim=True)
    max_scores_broadcast = max_scores.expand_as(vec)
    return max_scores + torch.log(torch.sum(torch.exp(vec - max_scores_broadcast), dim, keepdim=True))


# Mapped
def tensor_and_lengths(inputs):
    if isinstance(inputs, (list, tuple)):
        in_tensor, lengths = inputs
    else:
        in_tensor = inputs
        lengths = None  ##tf.reduce_sum(tf.cast(tf.not_equal(inputs, 0), tf.int32), axis=1)

    return in_tensor, lengths


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


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs

# Mapped
def get_activation(name="relu"):
    if name is None or name == "ident":
        return ident
    if name == "tanh":
        return nn.Tanh()
    if name == "hardtanh":
        return nn.Hardtanh()
    if name == 'leaky_relu':
        return nn.LeakyReLU()
    if name == "prelu":
        return nn.PReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "log_sigmoid":
        return nn.LogSigmoid()
    if name == "log_softmax":
        return nn.LogSoftmax(dim=-1)
    if name == "softmax":
        return nn.Softmax(dim=-1)
    return nn.ReLU()


def _cat_dir(h):
    return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], dim=-1)


def rnn_ident(output, hidden):
    return output, hidden


def rnn_signal(output, _):
    return output

def rnn_hidden(_, output_state):
    return output_state

def rnn_bi_hidden(output, output_state):
    if isinstance(output_state, tuple):
        output_state = tuple(_cat_dir(h) for h in output_state)
    else:
        output_state = _cat_dir(output_state)
    return output, output_state

#def pytorch_rnn(insz, hsz, rnntype, nlayers, dropout):
#    if nlayers == 1:
#        dropout = 0.0
#
#    if rnntype == 'gru':
#        rnn = torch.nn.GRU(insz, hsz, nlayers, dropout=dropout)
#    elif rnntype == 'blstm':
#        rnn = torch.nn.LSTM(insz, hsz//2, nlayers, dropout=dropout, bidirectional=True)
#        rnn = BiRNNWrapper(rnn, nlayers)
#    elif rnntype == 'bgru':
#        rnn = torch.nn.GRU(insz, hsz//2, nlayers, dropout=dropout, bidirectional=True)
#        rnn = BiRNNWrapper(rnn, nlayers)
#    else:
#        rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=dropout)
#    return rnn

# Mapped
class ConvEncoder(nn.Module):
    def __init__(self, insz, outsz, filtsz, pdrop, activation='relu'):
        super(ConvEncoder, self).__init__()
        self.output_dim = outsz
        pad = filtsz//2
        self.conv = nn.Conv1d(insz, outsz, filtsz, padding=pad)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, input_bct):
        conv_out = self.act(self.conv(input_bct))
        return self.dropout(conv_out)


# Mapped
class ConvEncoderStack(nn.Module):

    def __init__(self, insz, outsz, filtsz, pdrop, layers=1, activation='relu'):
        super(ConvEncoderStack, self).__init__()

        first_layer = ConvEncoder(insz, outsz, filtsz, pdrop, activation)
        subsequent_layer = ResidualBlock(ConvEncoder(outsz, outsz, filtsz, pdrop, activation))
        self.layers = nn.ModuleList([first_layer] + [copy.deepcopy(subsequent_layer) for _ in range(layers-1)])
        self.output_dim = outsz

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def bth2bht(t):
    return t.transpose(1, 2).contiguous()


def ident(t):
    return t


def tbh2bht(t):
    return t.permute(1, 2, 0).contiguous()

def tbh2bht(t):
    return t.transpose(0, 1).contiguous()


def bth2tbh(t):
    return t.transpose(0, 1).contiguous()

# Mapped
class ParallelConv(nn.Module):

    def __init__(self, insz, outsz, filtsz, activation='relu', input_fmt="bth"):
        super(ParallelConv, self).__init__()
        convs = []
        outsz_filts = outsz
        input_fmt = input_fmt.lower()
        if input_fmt == 'bth' or input_fmt == 'btc':
            self.transform_input = bth2bht
        elif input_fmt == 'tbh' or input_fmt == 'tbc':
            self.transform_input = tbh2bht
        else:
            self.transform_input = ident
        if type(outsz) == int:
            outsz_filts = len(filtsz) * [outsz]

        self.output_dim = sum(outsz_filts)
        for i, fsz in enumerate(filtsz):
            pad = fsz//2
            conv = nn.Sequential(
                nn.Conv1d(insz, outsz_filts[i], fsz, padding=pad),
                get_activation(activation)
            )
            convs.append(conv)
            # Add the module so its managed correctly
        self.convs = nn.ModuleList(convs)

    def forward(self, inputs):
        # TODO: change the input to btc?
        mots = []

        input_bct = self.transform_input(inputs)
        for conv in self.convs:
            # In Conv1d, data BxCxT, max over time
            conv_out = conv(input_bct)
            mot, _ = conv_out.max(2)
            mots.append(mot)
        mots = torch.cat(mots, 1)
        return mots # self.conv_drop(mots)

    @property
    def requires_length(self):
        return False

# Mapped
class Highway(nn.Module):

    def __init__(self, input_size, **kwargs):
        super(Highway, self).__init__()
        self.proj = nn.Linear(input_size, input_size)
        self.transform = nn.Linear(input_size, input_size)
        self.transform.bias.data.fill_(-2.0)
        self.output_dim = input_size

    def forward(self, input):
        proj_result = nn.functional.relu(self.proj(input))
        proj_gate = nn.functional.sigmoid(self.transform(input))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * input)
        return gated


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


class Dense(nn.Module):

    def __init__(self, insz, outsz, activation=None, unif=0, initializer=None):
        super(Dense, self).__init__()
        self.layer = pytorch_linear(insz, outsz, unif, initializer)
        self.activation = get_activation(activation)
        self.output_dim = outsz

    def forward(self, input):
        return self.activation(self.layer(input))


# Mapped
class ResidualBlock(nn.Module):

    def __init__(self, layer=None, **kwargs):
        super(ResidualBlock, self).__init__()
        self.layer = layer
        if self.layer is not None and hasattr(layer, 'output_dim'):
            self.output_dim = layer.output_dim

    def forward(self, input):
        return input + self.layer(input)


# Mapped
class SkipConnection(ResidualBlock):

    def __init__(self, input_size, activation='relu'):
        super(SkipConnection, self).__init__(None)
        self.layer = Dense(input_size, input_size, activation=activation)
        self.output_dim = input_size

# Mapped
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
        self.output_dim = num_features

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


# Mapped
class LSTMEncoder(nn.Module):

    def __init__(self, insz, hsz, nlayers, pdrop=0.0, output_fn=None, requires_length=True, batch_first=False, unif=0, initializer=None, **kwargs):
        """Produce a stack of biLSTMs with dropout performed on all but the last layer.

        :param insz: (``int``) The size of the input
        :param hsz: (``int``) The number of hidden units per LSTM
        :param nlayers: (``int``) The number of layers of LSTMs to stack
        :param pdrop: (``float``) The probability of dropping a unit value during dropout
        :param output_fn: function to determine what is returned from the encoder
        :param requires_length: (``bool``) Does this encoder require an input length in its inputs (defaults to ``True``)
        :param batch_first: (``bool``) Should we do batch first input or time-first input?
        :return: a stacked cell
        """
        super(LSTMEncoder, self).__init__()
        self._requires_length = requires_length
        if nlayers == 1:
            pdrop = 0.0
        self.rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=pdrop, bidirectional=False, batch_first=batch_first)
        if initializer == "ortho":
            nn.init.orthogonal(self.rnn.weight_hh_l0)
            nn.init.orthogonal(self.rnn.weight_ih_l0)
        elif initializer == "he" or initializer == "kaiming":
            nn.init.kaiming_uniform(self.rnn.weight_hh_l0)
            nn.init.kaiming_uniform(self.rnn.weight_ih_l0)
        elif unif > 0:
            for weight in self.rnn.parameters():
                weight.data.uniform_(-unif, unif)
        else:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        self.output_fn = rnn_ident if output_fn is None else output_fn
        self.output_dim = hsz

    def forward(self, inputs):
        tbc, lengths = tensor_and_lengths(inputs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths.tolist())
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        return self.output_fn(output, hidden)

    @property
    def requires_length(self):
        return self._requires_length


# Mapped
class BiLSTMEncoder(nn.Module):

    def __init__(self, insz, hsz, nlayers, pdrop=0.0, output_fn=None, requires_length=True, batch_first=False, unif=0, initializer=None, **kwargs):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param insz: (``int``) The size of the input
        :param hsz: (``int``) The number of hidden units per biLSTM (`hsz//2` used for each dir)
        :param nlayers: (``int``) The number of layers of LSTMs to stack
        :param dropout: (``int``) The probability of dropping a unit value during dropout
        :param output_fn: function to determine what is returned from the encoder
        :param requires_length: (``bool``) Does this encoder require an input length in its inputs (defaults to ``True``)
        :param batch_first: (``bool``) Should we do batch first input or time-first input?
        :return: a stacked cell
        """
        super(BiLSTMEncoder, self).__init__()
        self._requires_length = requires_length
        if nlayers == 1:
            pdrop = 0.0
        self.rnn = torch.nn.LSTM(insz, hsz//2, nlayers, dropout=pdrop, bidirectional=True, batch_first=batch_first)
        if initializer == "ortho":
            nn.init.orthogonal(self.rnn.weight_hh_l0)
            nn.init.orthogonal(self.rnn.weight_ih_l0)
        elif initializer == "he" or initializer == "kaiming":
            nn.init.kaiming_uniform(self.rnn.weight_hh_l0)
            nn.init.kaiming_uniform(self.rnn.weight_ih_l0)
        elif unif > 0:
            for weight in self.rnn.parameters():
                weight.data.uniform_(-unif, unif)
        else:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        self.output_fn = rnn_ident if output_fn is None else output_fn
        self.output_dim = hsz

    def forward(self, inputs):
        tbc, lengths = tensor_and_lengths(inputs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths.tolist())
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        return self.output_fn(output, hidden)

    @property
    def requires_length(self):
        return self._requires_length


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


class EmbeddingsStack(nn.Module):

    def __init__(self, embeddings_dict, dropout_rate=0.0, requires_length=False, **kwargs):
        """Takes in a dictionary where the keys are the input tensor names, and the values are the embeddings

        :param embeddings_dict: (``dict``) dictionary of each feature embedding
        """

        super(EmbeddingsStack, self).__init__()

        self.embeddings = EmbeddingsContainer()
        #input_sz = 0
        for k, embedding in embeddings_dict.items():
            self.embeddings[k] = embedding
            #input_sz += embedding.get_dsz()

        self.dropout = nn.Dropout(dropout_rate)
        self._requires_length = requires_length

    def cuda(self, device=None):
        super(EmbeddingsStack, self).cuda(device=device)
        for emb in self.embeddings.values():
            emb.cuda(device)

    def forward(self, inputs):
        """This method performs "embedding" of the inputs.  The base method here then concatenates along depth
        dimension to form word embeddings

        :return: A 3-d vector where the last dimension is the concatenated dimensions of all embeddings
        """
        all_embeddings_out = []
        for k, embedding in self.embeddings.items():
            x = inputs[k]
            embeddings_out = embedding(x)
            all_embeddings_out.append(embeddings_out)
        word_embeddings = torch.cat(all_embeddings_out, -1)
        return self.dropout(word_embeddings)

    @property
    def dsz(self):
        total_dsz = 0
        for embeddings in self.embeddings.values():
            total_dsz += embeddings.get_dsz()
        return total_dsz

    @property
    def output_dim(self):
        return self.dsz

    @property
    def requires_length(self):
        return self.requires_length


class DenseStack(nn.Module):

    def __init__(self, insz, hsz, activation='relu', pdrop_value=0.5, init=None, **kwargs):
        """Stack 1 or more hidden layers, optionally (forming an MLP)

        :param hsz: (``int``) The number of hidden units
        :param activation:  (``str``) The name of the activation function to use
        :param pdrop_value: (``float``) The dropout probability
        :param init: The tensorflow initializer

        """
        super(DenseStack, self).__init__()
        hszs = listify(hsz)
        self.output_dim = hsz[-1]
        self.layer_stack = nn.Sequential()
        current = insz
        for i, hsz in hszs:
            self.layer_stack.append(WithDropout(Dense(current, hsz, activation=activation), pdrop_value))
            current = hsz

    def forward(self, inputs):
        """Stack 1 or more hidden layers, optionally (forming an MLP)

        :param inputs: The fixed representation of the model
        :param training: (``bool``) A boolean specifying if we are training or not
        :param init: The tensorflow initializer
        :param kwargs: See below

        :Keyword Arguments:
        * *hsz* -- (``int``) The number of hidden units (defaults to `100`)

        :return: The final layer
        """
        x = inputs
        for layer in self.layer_stack:
            x = layer(x)
        return x

    @property
    def requires_length(self):
        return False


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


class FineTuneModel(nn.Module):

    def __init__(self, nc, embeddings, stack_model=None):
        super(FineTuneModel, self).__init__()
        if isinstance(embeddings, dict):
            self.finetuned = EmbeddingsStack(embeddings)
        else:
            self.finetuned = embeddings
        self.stack_model = stack_model
        output_dim = self.finetuned.get_dsz() if stack_model is None else stack_model.output_dim
        self.output_layer = Dense(output_dim, nc, activation="log_softmax")

    def forward(self, inputs):
        base_layers = self.finetuned(inputs)
        stacked = self.stack_model(base_layers) if self.stack_model is not None else base_layers
        return self.output_layer(stacked)


class EmbedPoolStackModel(nn.Module):

    def __init__(self, nc, embeddings, pool_model, stack_model=None):
        super(EmbedPoolStackModel, self).__init__()
        if isinstance(embeddings, dict):
            self.embed_model = EmbeddingsStack(embeddings)
        else:
            self.embed_model = embeddings

        self.pool_requires_length = False
        if hasattr(pool_model, 'requires_length'):
            self.pool_requires_length = pool_model.requires_length

        self.pool_model = pool_model
        output_dim = self.pool_model.output_dim if stack_model is None else stack_model.output_dim
        self.output_layer = Dense(output_dim, nc, activation="log_softmax")
        self.stack_model = stack_model

    def forward(self, inputs):
        lengths = inputs.get('lengths')

        embedded = self.embed_model(inputs)

        if self.pool_requires_length:
            embedded = (embedded, lengths)
        pooled = self.pool_model(embedded)
        stacked = self.stack_model(pooled) if self.stack_model is not None else pooled
        return self.output_layer(stacked)

    #def cuda(self, device=None):
    #    super(EmbedPoolStackModel, self).cuda(device=device)
    #    self.embed_model.cuda(device=device)


class WithDropout(nn.Module):

    def __init__(self, layer, pdrop=0.5):
        super(WithDropout, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(pdrop)

    def forward(self, inputs):
        return self.dropout(self.layer(inputs))

    @property
    def output_dim(self):
        return self.layer.output_dim


def transition_mask(vocab, span_type, s_idx, e_idx, pad_idx=None):
    """Create a mask to enforce span sequence transition constraints.

    Returns a Tensor with valid transitions as a 0 and invalid as a 1 for easy use with `masked_fill`
    """
    np_mask = transition_mask_np(vocab, span_type, s_idx, e_idx, pad_idx=pad_idx)
    return torch.from_numpy(np_mask) == 0


def viterbi(unary, trans, lengths, start_idx, end_idx, norm=False):
    """Do Viterbi decode on a batch.

    :param unary: torch.FloatTensor: [T, B, N]
    :param trans: torch.FloatTensor: [1, N, N]

    :return: List[torch.LongTensor]: [[T] .. B] that paths
    :return: torch.FloatTensor: [B] the path scores
    """
    seq_len, batch_size, tag_size = unary.size()
    min_length = torch.min(lengths)
    batch_range = torch.arange(batch_size, dtype=torch.int64)
    backpointers = []

    # Alphas: [B, 1, N]
    alphas = torch.Tensor(batch_size, 1, tag_size).fill_(-1e4).to(unary.device)
    alphas[:, 0, start_idx] = 0
    alphas = F.log_softmax(alphas, dim=-1) if norm else alphas

    for i, unary_t in enumerate(unary):
        next_tag_var = alphas + trans
        viterbi, best_tag_ids = torch.max(next_tag_var, 2)
        backpointers.append(best_tag_ids.data)
        new_alphas = viterbi + unary_t
        new_alphas.unsqueeze_(1)
        if i >= min_length:
            mask = (i < lengths).view(-1, 1, 1)
            alphas = alphas.masked_fill(mask, 0) + new_alphas.masked_fill(mask == 0, 0)
        else:
            alphas = new_alphas

    # Add end tag
    terminal_var = alphas.squeeze(1) + trans[:, end_idx]
    _, best_tag_id = torch.max(terminal_var, 1)
    path_score = terminal_var[batch_range, best_tag_id]  # Select best_tag from each batch

    best_path = [best_tag_id]
    # Flip lengths
    rev_len = seq_len - lengths - 1
    for i, backpointer_t in enumerate(reversed(backpointers)):
        # Get new best tag candidate
        new_best_tag_id = backpointer_t[batch_range, best_tag_id]
        # We are going backwards now, if you passed your flipped length then you aren't in your real results yet
        mask = (i > rev_len)
        best_tag_id = best_tag_id.masked_fill(mask, 0) + new_best_tag_id.masked_fill(mask == 0, 0)
        best_path.append(best_tag_id)
    _ = best_path.pop()
    best_path.reverse()
    best_path = torch.stack(best_path)
    # Return list of paths
    paths = []
    best_path = best_path.transpose(0, 1)
    for path, length in zip(best_path, lengths):
        paths.append(path[:length])
    return paths, path_score.squeeze(0)


class TaggerGreedyDecoder(nn.Module):

    def __init__(self, num_tags, constraint_mask=None, batch_first=True):
        super(TaggerGreedyDecoder, self).__init__()
        self.num_tags = num_tags

        if constraint_mask is not None:
            self.mask = F.log_softmax(torch.zeros(constraint_mask.shape).masked_fill(constraint_mask, -1e4), dim=0)
            self.register_buffer('constraint', self.mask.unsqueeze(0))
        self.to_batch_first = ident if batch_first else tbh2bht
        self.to_time_first = bth2tbh if batch_first else ident

    @property
    def transitions(self):
        return self.mask

    def neg_log_loss(self, inputs, tags, lengths):
        # Cross entropy loss
        unaries = self.to_batch_first(inputs)
        loss = F.cross_entropy(unaries, tags, size_average=False, ignore_index=Offsets.PAD)
        batch_size = inputs.size()[0]
        loss /= batch_size
        return loss

    def call(self, inputs):

        unaries, lengths = inputs
        if self.constraint is not None:
            probv = self.to_time_first(unaries)
            probv = F.log_softmax(probv, dim=-1)
            preds, _ = viterbi(probv, self.constraint, lengths, Offsets.GO, Offsets.EOS, norm=True)
        else:
            probv = self.to_batch_first(unaries)
            preds = []
            for pij, sl in zip(probv, lengths):
                _, unary = torch.max(pij[:sl], 1)
                preds.append(unary.data)
        return preds


class CRF(nn.Module):

    def __init__(self, num_tags, constraint_mask=None, batch_first=True, idxs=(Offsets.GO, Offsets.EOS)):
        """Initialize the object.
        :param num_tags: int, The number of tags in your output (emission size)
        :param idxs: Tuple(int. int), The index of the start and stop symbol
            in emissions.
        :param batch_first: bool, if the input [B, T, ...] or [T, B, ...]
        :param mask: torch.ByteTensor, Constraints on the transitions [1, N, N]

        Note:
            if idxs is none then the CRF adds these symbols to the emission
            vectors and n_tags is assumed to be the number of output tags.
            if idxs is not none then the first element is assumed to be the
            start index and the second idx is assumed to be the end index. In
            this case n_tags is assumed to include the start and end symbols.
        """
        super(CRF, self).__init__()
        self.start_idx, self.end_idx = idxs
        self.num_tags = num_tags
        if constraint_mask is not None:
            self.register_buffer('mask', constraint_mask)
        else:
            self.mask = None

        self.transitions_p = nn.Parameter(torch.Tensor(1, self.num_tags, self.num_tags).zero_())
        self.batch_first = batch_first

    def extra_repr(self):
        str_ = "n_tags=%d, batch_first=%s" % (self.num_tags, self.batch_first)
        if self.mask is not None:
            str_ += ", masked=True"
        return str_

    @property
    def transitions(self):
        if self.mask is not None:
            return self.transitions_p.masked_fill(self.mask, -1e4)
        return self.transitions_p

    def neg_log_loss(self, unary, tags, lengths):
        """Neg Log Loss with a Batched CRF.

        :param unary: torch.FloatTensor: [T, B, N] or [B, T, N]
        :param tags: torch.LongTensor: [T, B] or [B, T]
        :param lengths: torch.LongTensor: [B]

        :return: torch.FloatTensor: [B]
        """
        # Convert from [B, T, N] -> [T, B, N]
        if self.batch_first:
            unary = unary.transpose(0, 1)
            tags = tags.transpose(0, 1)
        _, batch_size, _ = unary.size()
        fwd_score = self.forward_alg(unary, lengths)  # TODO: shouldnt this call __call__?
        gold_score = self.score_sentence(unary, tags, lengths)

        loss = fwd_score - gold_score
        batch_loss = torch.mean(loss)
        return batch_loss

    def score_sentence(self, unary, tags, lengths):
        """Score a batch of sentences.

        :param unary: torch.FloatTensor: [T, B, N]
        :param tags: torch.LongTensor: [T, B]
        :param lengths: torch.LongTensor: [B]
        :param batzh_size: int: B
        :param min_length: torch.LongTensor: []

        :return: torch.FloatTensor: [B]
        """
        min_length = torch.min(lengths)
        batch_size = lengths.shape[0]
        trans = self.transitions.squeeze(0)  # [N, N]
        batch_range = torch.arange(batch_size, dtype=torch.int64)  # [B]
        start = torch.full((1, batch_size), self.start_idx, dtype=tags.dtype, device=tags.device)  # [1, B]
        tags = torch.cat([start, tags], 0)  # [T, B]
        scores = torch.zeros(batch_size, requires_grad=True).to(unary.device)  # [B]
        for i, unary_t in enumerate(unary):
            new_scores = (
                trans[tags[i + 1], tags[i]] +
                unary_t[batch_range, tags[i + 1]]
            )
            if i >= min_length:
                # If we are farther along `T` than your length don't add to your score
                mask = (i >= lengths)
                scores = scores + new_scores.masked_fill(mask, 0)
            else:
                scores = scores + new_scores
        # Add stop tag
        scores = scores + trans[self.end_idx, tags[lengths, batch_range]]
        return scores

    def forward_alg(self, unary, lengths):
        """For CRF forward on a batch.

        :param unary: torch.FloatTensor: [T, B, N]
        :param lengths: torch.LongTensor: [B]

        :return: torch.FloatTensor: [B]
        """
        # alphas: [B, 1, N]
        min_length = torch.min(lengths)
        batch_size = lengths.shape[0]
        alphas = torch.Tensor(batch_size, 1, self.num_tags).fill_(-1e4).to(unary.device)
        alphas[:, 0, self.start_idx] = 0.
        alphas.requires_grad = True

        trans = self.transitions  # [1, N, N]

        for i, unary_t in enumerate(unary):
            # unary_t: [B, N]
            unary_t = unary_t.unsqueeze(2)  # [B, N, 1]
            # Broadcast alphas along the rows of trans
            # Broadcast trans along the batch of alphas
            # [B, 1, N] + [1, N, N] -> [B, N, N]
            # Broadcast unary_t along the cols of result
            # [B, N, N] + [B, N, 1] -> [B, N, N]
            scores = alphas + trans + unary_t
            new_alphas = vec_log_sum_exp(scores, 2).transpose(1, 2)
            # If we haven't reached your length zero out old alpha and take new one.
            # If we are past your length, zero out new_alpha and keep old one.

            if i >= min_length:
                mask = (i < lengths).view(-1, 1, 1)
                alphas = alphas.masked_fill(mask, 0) + new_alphas.masked_fill(mask == 0, 0)
            else:
                alphas = new_alphas

        terminal_vars = alphas + trans[:, self.end_idx]
        alphas = vec_log_sum_exp(terminal_vars, 2)
        return alphas.squeeze()

    def forward(self, inputs):
        unary, lengths = inputs
        if self.training:
            return self.forward_alg(unary, lengths)

        with torch.no_grad():
            return self.decode(unary, lengths)[0]

    def decode(self, unary, lengths):
        """Do Viterbi decode on a batch.

        :param unary: torch.FloatTensor: [T, B, N] or [B, T, N]
        :param lengths: torch.LongTensor: [B]

        :return: List[torch.LongTensor]: [B] the paths
        :return: torch.FloatTensor: [B] the path score
        """
        if self.batch_first:
            unary = unary.transpose(0, 1)
        trans = self.transitions  # [1, N, N]
        return viterbi(unary, trans, lengths, self.start_idx, self.end_idx)


class SequenceModel(nn.Module):

    def __init__(self, nc, embeddings, transducer, decoder=None):
        super(SequenceModel, self).__init__()
        if isinstance(embeddings, dict):
            self.embed_model = EmbeddingsStack(embeddings)
        else:
            self.embed_model = embeddings
        self.transducer_model = transducer
        self.proj_layer = Dense(transducer.output_dim, nc)
        self.decoder_model = decoder

    def transduce(self, inputs):
        lengths = inputs.get('lengths')

        embedded = self.embed_model(inputs)
        embedded = (embedded, lengths)
        transduced = self.proj_layer(self.transducer_model(embedded))
        return transduced

    def decode(self, transduced, lengths):
        return self.decoder_model((transduced, lengths))

    def forward(self, inputs):
        transduced = self.transduce(inputs)
        return self.decode(transduced, inputs.get('lengths'))


class TagSequenceModel(SequenceModel):

    def __init__(self, nc, embeddings, transducer, decoder=None):
        decoder_model = CRF(nc) if decoder is None else decoder
        super(TagSequenceModel, self).__init__(nc, embeddings, transducer, decoder_model)

    def neg_log_loss(self, unary, tags, lengths):
        return self.decoder_model.neg_log_loss(unary, tags, lengths)

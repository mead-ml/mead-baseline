import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from eight_mile.utils import listify, Offsets
from eight_mile.utils import transition_mask as transition_mask_np
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
        super().__init__()
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


class SequenceLoss(nn.Module):

    def __init__(self, LossFn=nn.NLLLoss, avg='token'):
        """A class that applies a Loss function to sequence via the folding trick."""
        super().__init__()
        self.avg = avg
        if avg == 'token':
            self.crit = LossFn(ignore_index=Offsets.PAD, reduction='mean')
            self._norm = self._no_norm
        else:
            self.crit = LossFn(ignore_index=Offsets.PAD, reduction='sum')
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

    def extra_repr(self):
        return f"reduction={self.avg}"


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inputs):
        return inputs


class MeanPool1D(nn.Module):
    def __init__(self, outsz, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.reduction_dim = 1 if self.batch_first else 0
        self.output_dim = outsz

    def forward(self, inputs):
        tensor, lengths = tensor_and_lengths(inputs)
        # Regardless of whether the input is `[B, T, H]` or `[T, B, H]` the shape after
        # the sum is `[B, H]` so the lengths (of shape `[B]`) should be unsqueezed to
        # `[B, 1]` in order to broadcast
        return torch.sum(tensor, self.reduction_dim, keepdim=False) / torch.unsqueeze(lengths, -1).to(tensor.dtype).to(tensor.device)

    @property
    def requires_length(self):
        return True

    def extra_repr(self):
        return f"batch_first={self.batch_first}"


class MaxPool1D(nn.Module):
    def __init__(self, outsz, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.reduction_dim = 1 if self.batch_first else 0
        self.output_dim = outsz

    def forward(self, inputs):
        tensor, lengths = tensor_and_lengths(inputs)
        if lengths is not None:
            # If tensor = `[B, T, H]`
            #    mask = `[B, T, 1]`
            # If tensor = `[T, B, H]`
            #    mask = `[T, B, 1]`
            # So it will mask all the values in H past the right length
            mask = sequence_mask(lengths).to(tensor.device)
            mask = mask if self.batch_first else bth2tbh(mask)
            # Fill masked with very negative so it never gets selected
            tensor = tensor.masked_fill(mask.unsqueeze(-1) == 0, -1e4)
        dmax, _ = torch.max(tensor, self.reduction_dim, keepdim=False)
        return dmax

    def extra_repr(self):
        return f"batch_first={self.batch_first}"


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
    """Concat forward and backword state vectors.

    The shape of the hidden is `[#layers * #dirs, B, H]`. The docs say you can
    separate directions with `h.view(#l, #dirs, B, H)` with the forward dir being
    index 0 and backwards dir being 1.

    This means that before separating with the view the forward dir are the even
    indices in the first dim while the backwards dirs are the odd ones. Here we select
    the even and odd values and concatenate them
    """
    return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], dim=-1)


def concat_state_dirs(state):
    """Convert the bidirectional out of an RNN so the forward and backward values are a single vector."""
    if isinstance(state, tuple):
        return tuple(_cat_dir(h) for h in state)
    return _cat_dir(state)


# Mapped
class ConvEncoder(nn.Module):
    def __init__(self, insz, outsz, filtsz, pdrop, activation='relu'):
        super().__init__()
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
        super().__init__()

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


def tbh2bth(t):
    return t.transpose(0, 1).contiguous()


def bth2tbh(t):
    return t.transpose(0, 1).contiguous()


# Mapped
class ParallelConv(nn.Module):

    def __init__(self, insz, outsz, filtsz, activation='relu', input_fmt="bth"):
        super().__init__()
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
        super().__init__()
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


class StackedLSTMCell(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super().__init__()
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
        super().__init__()
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



class Dense(nn.Module):

    def __init__(self, insz, outsz, activation=None, unif=0, initializer=None):
        super().__init__()
        self.layer = pytorch_linear(insz, outsz, unif, initializer)
        self.activation = get_activation(activation)
        self.output_dim = outsz

    def forward(self, input):
        return self.activation(self.layer(input))


class WeightTieDense(nn.Module):
    def __init__(self, tie):
        super().__init__()
        self.tie = tie
        self.weight, self.transform = self._get_weight(tie)
        self.register_parameter('bias', None)

    def _get_weight(self, tie):
        emb = getattr(tie, 'embeddings', None)
        if emb is not None:
            return getattr(emb, 'weight'), self._identity
        return getattr(tie, 'weight'), self._transpose

    def _identity(self, x):
        return x

    def _transpose(self, x):
        return x.transpose(0, 1).contiguous()

    def forward(self, input):
        return F.linear(input, self.transform(self.weight), self.bias)



# Mapped
class ResidualBlock(nn.Module):

    def __init__(self, layer=None, **kwargs):
        super().__init__()
        self.layer = layer
        if self.layer is not None and hasattr(layer, 'output_dim'):
            self.output_dim = layer.output_dim

    def forward(self, input):
        return input + self.layer(input)


# Mapped
class SkipConnection(ResidualBlock):

    def __init__(self, input_size, activation='relu'):
        super().__init__(None)
        self.layer = Dense(input_size, input_size, activation=activation)
        self.output_dim = input_size


def rnn_cell(insz, hsz, rnntype, nlayers, dropout):

    if rnntype == 'gru':
        rnn = StackedGRUCell(nlayers, insz, hsz, dropout)
    else:
        rnn = StackedLSTMCell(nlayers, insz, hsz, dropout)
    return rnn

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


# Obnoxious, now that TF has a deferred mode, the insz isnt required but it is for this one.  Should we make it
# the same for TF?
class LSTMEncoder(nn.Module):

    def __init__(self, insz, hsz, nlayers, pdrop=0.0, requires_length=True, batch_first=False, unif=0, initializer=None, **kwargs):
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
        super().__init__()
        self._requires_length = requires_length
        self.batch_first = batch_first
        self.nlayers = nlayers
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
        self.output_dim = hsz

    def forward(self, inputs):
        tbc, lengths = tensor_and_lengths(inputs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths.tolist(), batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return self.output_fn(output, hidden)

    @property
    def requires_length(self):
        return self._requires_length

    def output_fn(self, output, state):
        return output, self.extract_top_state(state)

    def extract_top_state(self, state):
        # Select the topmost state with -1 and the only direction is forward (select with 0)
        return tuple(s.view(self.nlayers, 1, -1, self.output_dim)[-1, 0] for s in state)


class LSTMEncoderSequence(LSTMEncoder):

    def output_fn(self, output, state):
        return output


class LSTMEncoderWithState(nn.Module):


    def __init__(self, insz, hsz, nlayers, pdrop=0.0, batch_first=False, unif=0, initializer=None, **kwargs):
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
        super().__init__()
        self._requires_length = False
        self.batch_first = batch_first
        self.nlayers = nlayers
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
        self.output_dim = hsz

    @property
    def requires_state(self):
        return True

    def forward(self, inputs):
        inputs, hidden = inputs
        output, hidden = self.rnn(inputs, hidden)
        return output, hidden  ##concat_state_dirs(hidden)

    @property
    def requires_length(self):
        return self._requires_length


class LSTMEncoderAll(LSTMEncoder):

    def output_fn(self, output, state):
        return output, state


class LSTMEncoderHidden(LSTMEncoder):

    def output_fn(self, output, state):
        return self.extract_top_state(state)[0]


class LSTMEncoderHiddenContext(LSTMEncoder):
    def output_fn(self, output, state):
        return self.extract_top_state(state)


class BiLSTMEncoder(nn.Module):

    def __init__(self, insz, hsz, nlayers, pdrop=0.0, requires_length=True, batch_first=False, unif=0, initializer=None, **kwargs):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param insz: (``int``) The size of the input
        :param hsz: (``int``) The number of hidden units per biLSTM (`hsz//2` used for each dir)
        :param nlayers: (``int``) The number of layers of LSTMs to stack
        :param dropout: (``int``) The probability of dropping a unit value during dropout
        :param requires_length: (``bool``) Does this encoder require an input length in its inputs (defaults to ``True``)
        :param batch_first: (``bool``) Should we do batch first input or time-first input?
        :return: a stacked cell
        """
        super().__init__()
        self._requires_length = requires_length
        self.batch_first = batch_first
        self.nlayers = nlayers
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
        self.output_dim = hsz

    def forward(self, inputs):
        tbc, lengths = tensor_and_lengths(inputs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths.tolist(), batch_first=self.batch_first)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        return self.output_fn(output, hidden)

    def extract_top_state(self, state):
        # Select the topmost state with -1 and the only direction is forward (select with 0)
        return tuple(s.view(self.nlayers, 1, -1, self.output_dim)[-1, 0] for s in state)

    def output_fn(self, output, state):
        return output, self.extract_top_state(concat_state_dirs(state))

    @property
    def requires_length(self):
        return self._requires_length


class BiLSTMEncoderAll(BiLSTMEncoder):
    def output_fn(self, output, state):
        return output, concat_state_dirs(state)


class BiLSTMEncoderSequence(BiLSTMEncoder):

    def output_fn(self, output, state):
        return output


class BiLSTMEncoderHidden(BiLSTMEncoder):

    def output_fn(self, output, state):
        return self.extract_top_state(concat_state_dirs(state))[0]


class BiLSTMEncoderHiddenContext(BiLSTMEncoder):

    def output_fn(self, output, state):
        return self.extract_top_state(concat_state_dirs(state))


class EmbeddingsContainer(nn.Module):
    def __init__(self):
        super().__init__()

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

        super().__init__()

        self.embeddings = EmbeddingsContainer()
        #input_sz = 0
        for k, embedding in embeddings_dict.items():
            self.embeddings[k] = embedding
            #input_sz += embedding.get_dsz()

        self.dropout = nn.Dropout(dropout_rate)
        self._requires_length = requires_length

    def cuda(self, device=None):
        super().cuda(device=device)
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

        :param insz: (``int``) The number of input units
        :param hsz: (``int``) The number of hidden units
        :param activation:  (``str``) The name of the activation function to use
        :param pdrop_value: (``float``) The dropout probability
        :param init: The tensorflow initializer

        """
        super().__init__()
        hszs = listify(hsz)
        self.output_dim = hsz[-1]
        current = insz
        layer_stack = []
        for hsz in hszs:
            layer_stack.append(WithDropout(Dense(current, hsz, activation=activation), pdrop_value))
            current = hsz
        self.layer_stack = nn.Sequential(*layer_stack)

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
        super().__init__()
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
        attended = torch.tanh(self.W_c(attended))
        return attended


class LuongDotProductAttention(BaseAttention):

    def __init__(self, hsz):
        super().__init__(hsz)

    def _attention(self, query_t, keys_bth, keys_mask):
        a = keys_bth @ query_t.unsqueeze(2)
        a = a.squeeze(2).masked_fill(keys_mask == 0, -1e9)
        a = F.softmax(a, dim=-1)
        return a


class ScaledDotProductAttention(BaseAttention):

    def __init__(self, hsz):
        super().__init__(hsz)

    def _attention(self, query_t, keys_bth, keys_mask):
        a = (keys_bth @ query_t.unsqueeze(2)) / math.sqrt(self.hsz)
        a = a.squeeze(2).masked_fill(keys_mask == 0, -1e9)
        a = F.softmax(a, dim=-1)
        return a


class LuongGeneralAttention(BaseAttention):

    def __init__(self, hsz):
        super().__init__(hsz)
        self.W_a = nn.Linear(self.hsz, self.hsz, bias=False)

    def _attention(self, query_t, keys_bth, keys_mask):
        a = keys_bth @ self.W_a(query_t).unsqueeze(2)
        a = a.squeeze(2).masked_fill(keys_mask == 0, -1e9)
        a = F.softmax(a, dim=-1)
        return a


class BahdanauAttention(BaseAttention):

    def __init__(self, hsz):
        super().__init__(hsz)
        self.hsz = hsz
        self.W_a = nn.Linear(self.hsz, self.hsz, bias=False)
        self.E_a = nn.Linear(self.hsz, self.hsz, bias=False)
        self.v = nn.Linear(self.hsz, 1, bias=False)

    def _attention(self, query_t, keys_bth, keys_mask):
        B, T, H = keys_bth.shape
        q = self.W_a(query_t.view(-1, self.hsz)).view(B, 1, H)
        u = self.E_a(keys_bth).view(B, T, H)
        z = torch.tanh(q + u)
        a = self.v(z.view(-1, self.hsz)).view(B, T)
        a.masked_fill(keys_mask == 0, -1e9)
        a = F.softmax(a, dim=-1)
        return a

    def _update(self, a, query_t, values_bth):
        # a = B x T
        # Want to apply over context, scaled by a
        # (B x 1 x T) (B x T x H) = (B x 1 x H) -> (B x H)
        a = a.view(a.size(0), 1, a.size(1))
        c_t = (a @ values_bth).squeeze(1)
        # (B x 2H)
        attended = torch.cat([c_t, query_t], -1)
        attended = self.W_c(attended)
        return attended


class FineTuneModel(nn.Module):

    def __init__(self, nc, embeddings, stack_model=None):
        super().__init__()
        if isinstance(embeddings, dict):
            self.finetuned = EmbeddingsStack(embeddings)
        else:
            self.finetuned = embeddings
        self.stack_model = stack_model
        output_dim = self.finetuned.output_dim if stack_model is None else stack_model.output_dim
        self.output_layer = Dense(output_dim, nc, activation="log_softmax")

    def forward(self, inputs):
        base_layers = self.finetuned(inputs)
        stacked = self.stack_model(base_layers) if self.stack_model is not None else base_layers
        return self.output_layer(stacked)


class CompositePooling(nn.Module):
    def __init__(self, models):
        """
        Note, this currently requires that each submodel is an eight_mile model with an `output_dim` attr
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.output_dim = sum(m.output_dim for m in self.models)
        self._requires_length = any(getattr(m, 'requires_length', False) for m in self.models)

    def forward(self, inputs):
        inputs, lengths = tensor_and_lengths(inputs)
        pooled = []
        for sub_model in self.models:
            if getattr(sub_model, 'requires_length', False):
                pooled.append(sub_model((inputs, lengths)))
            else:
                pooled.append(sub_model(inputs))
        return torch.cat(pooled, -1)

    @property
    def requires_length(self):
        return self._requires_length


class EmbedPoolStackModel(nn.Module):

    def __init__(self, nc, embeddings, pool_model, stack_model=None):
        super().__init__()
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
        super().__init__()
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


def viterbi(unary, trans, lengths, start_idx, end_idx, norm = lambda x, y: x):
    """Do Viterbi decode on a batch.

    :param unary: torch.FloatTensor: [T, B, N]
    :param trans: torch.FloatTensor: [1, N, N]
    :param norm: Callable: This function should take the initial and a dim to
        normalize along.

    :return: torch.LongTensor: [T, B] the padded paths
    :return: torch.FloatTensor: [B] the path scores
    """
    seq_len, batch_size, tag_size = unary.size()
    min_length = torch.min(lengths)
    backpointers = []

    # Alphas: [B, 1, N]
    alphas = torch.Tensor(batch_size, 1, tag_size).fill_(-1e4).to(unary.device)
    alphas[:, 0, start_idx] = 0
    alphas = norm(alphas, -1) if norm else alphas

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
    terminal_var = alphas.squeeze(1) + trans[:, end_idx, :]
    path_score, best_tag_id = torch.max(terminal_var, 1)
    # Flip lengths
    rev_len = seq_len - lengths - 1

    best_path = [best_tag_id]
    for i, backpointer_t in enumerate(reversed(backpointers)):
        # Get new best tag candidate
        new_best_tag_id = backpointer_t.gather(1, best_tag_id.unsqueeze(1)).squeeze(1)
        # We are going backwards now, if flipped length was passed
        # these you aren't in your real results yet
        mask = (i > rev_len)
        best_tag_id = best_tag_id.masked_fill(mask, 0) + new_best_tag_id.masked_fill(mask == 0, 0)
        best_path.append(best_tag_id)
    _ = best_path.pop()
    best_path.reverse()
    best_path = torch.stack(best_path)
    # Mask out the extra tags (This might be pointless given that anything that
    # will use this as a dense tensor downstream will mask it itself?)
    seq_mask = sequence_mask(lengths).to(best_path.device).transpose(0, 1)
    best_path = best_path.masked_fill(seq_mask == 0, 0)
    return best_path, path_score


class TaggerGreedyDecoder(nn.Module):

    def __init__(self, num_tags, constraint_mask=None, batch_first=True, reduction='batch'):
        """A Greedy decoder and loss module for taggers.

        :param num_tags: `int` The number of output classes
        :param constraint_mask: `Tensor[1, N, N]` A mask with valid transitions as 1 and invalid as 0
        :param batch_first: `bool` Should the batch dimensions be first?
        :param reduction: `str` Should the loss be calculated at the token level or batch level
        """
        super().__init__()
        self.num_tags = num_tags

        if constraint_mask is not None:
            constraint_mask = F.log_softmax(torch.zeros(constraint_mask.shape).masked_fill(constraint_mask, -1e4), dim=1)
            self.register_buffer('constraint_mask', constraint_mask)
        else:
            self.constraint_mask = None
        self.to_batch_first = ident if batch_first else tbh2bth
        self.to_time_first = bth2tbh if batch_first else ident
        self.batch_first = batch_first
        self.loss = SequenceLoss(LossFn=nn.CrossEntropyLoss, avg=reduction)

    @property
    def transitions(self):
        return self.constraint_mask

    def neg_log_loss(self, inputs, tags, lengths):
        unaries = self.to_batch_first(inputs)
        tags = self.to_batch_first(tags)
        return self.loss(unaries, tags)

    def forward(self, inputs):
        unaries, lengths = tensor_and_lengths(inputs)
        # If there is a constraint mask do a masked viterbi
        if self.constraint_mask is not None:
            probv = self.to_time_first(unaries)
            probv = F.log_softmax(probv, dim=-1)
            preds, scores = viterbi(probv, self.constraint_mask, lengths, Offsets.GO, Offsets.EOS, norm=F.log_softmax)
            if self.batch_first:
                return tbh2bth(preds), scores
        else:
            # Decoding doesn't care about batch/time first
            _, preds = torch.max(unaries, -1)
            mask = sequence_mask(lengths).to(preds.device)
            # The mask gets generated as batch first
            mask = mask if self.batch_first else mask.transpose(0, 1)
            preds = preds.masked_fill(mask == 0, 0)
        return preds, None

    def extra_repr(self):
        str_ = f"n_tags={self.num_tags}, batch_first={self.batch_first}"
        if self.constraint_mask is not None:
            str_ += ", constrained=True"
        return str_


class CRF(nn.Module):

    def __init__(self, num_tags, constraint_mask=None, batch_first=True, idxs=(Offsets.GO, Offsets.EOS)):
        """Initialize the object.
        :param num_tags: int, The number of tags in your output (emission size)
        :param constraint: torch.ByteTensor, Constraints on the transitions [1, N, N]
        :param idxs: Tuple(int. int), The index of the start and stop symbol
            in emissions.
        :param batch_first: bool, if the input [B, T, ...] or [T, B, ...]

        Note:
            if idxs is none then the CRF adds these symbols to the emission
            vectors and n_tags is assumed to be the number of output tags.
            if idxs is not none then the first element is assumed to be the
            start index and the second idx is assumed to be the end index. In
            this case n_tags is assumed to include the start and end symbols.
        """
        super().__init__()
        self.start_idx, self.end_idx = idxs
        self.num_tags = num_tags
        if constraint_mask is not None:
            self.register_buffer('constraint_mask', constraint_mask)
        else:
            self.constraint_mask = None

        self.transitions_p = nn.Parameter(torch.Tensor(1, self.num_tags, self.num_tags).zero_())
        self.batch_first = batch_first

    def extra_repr(self):
        str_ = "n_tags=%d, batch_first=%s" % (self.num_tags, self.batch_first)
        if self.constraint_mask is not None:
            str_ += ", constrained=True"
        return str_

    @property
    def transitions(self):
        if self.constraint_mask is not None:
            return self.transitions_p.masked_fill(self.constraint_mask, -1e4)
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
        fwd_score = self._forward_alg(unary, lengths)
        gold_score = self.score_sentence(unary, tags, lengths)

        loss = fwd_score - gold_score
        batch_loss = torch.mean(loss)
        return batch_loss

    def score_sentence(self, unary, tags, lengths):
        """Score a batch of sentences.

        :param unary: torch.FloatTensor: [T, B, N]
        :param tags: torch.LongTensor: [T, B]
        :param lengths: torch.LongTensor: [B]
        :param min_length: torch.LongTensor: []

        :return: torch.FloatTensor: [B]
        """
        batch_size = lengths.shape[0]
        assert lengths.shape[0] == unary.shape[1]

        trans = self.transitions.squeeze(0)  # [N, N]
        start = torch.full((1, batch_size), self.start_idx, dtype=tags.dtype, device=tags.device)  # [1, B]
        tags = torch.cat([start, tags], 0)  # [T + 1, B]

        # Unfold gives me all slices of size 2 (this tag next tag) from dimension T
        tag_pairs = tags.unfold(0, 2, 1)
        # Move the pair dim to the front and split it into two
        indices = tag_pairs.permute(2, 0, 1).chunk(2)
        trans_score = trans[[indices[1], indices[0]]].squeeze(0)
        # Pull out the values of the tags from the unary scores.
        unary_score = unary.gather(2, tags[1:].unsqueeze(-1)).squeeze(-1)

        mask = sequence_mask(lengths).transpose(0, 1).to(tags.device)
        scores = unary_score + trans_score
        scores = scores.masked_fill(mask == 0, 0)
        scores = scores.sum(0)

        eos_scores = trans[self.end_idx, tags.gather(0, lengths.unsqueeze(0)).squeeze(0)]
        scores = scores + eos_scores
        return scores

    def _forward_alg(self, unary, lengths):
        """For CRF forward on a batch.

        :param unary: torch.FloatTensor: [T, B, N]
        :param lengths: torch.LongTensor: [B]

        :return: torch.FloatTensor: [B]
        """
        # alphas: [B, 1, N]
        min_length = torch.min(lengths)
        batch_size = lengths.shape[0]
        lengths.shape[0] == unary.shape[1]

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
        return alphas.view(batch_size)

    def forward(self, inputs):
        unary, lengths = inputs
        if self.training:
            if self.batch_first:
                unary = unary.transpose(0, 1)
            return self._forward_alg(unary, lengths)

        with torch.no_grad():
            return self.decode(unary, lengths)

    def decode(self, unary, lengths):
        """Do Viterbi decode on a batch.

        :param unary: torch.FloatTensor: [T, B, N] or [B, T, N]
        :param lengths: torch.LongTensor: [B]

        :return: torch.LongTensor: [B] the paths
        :return: torch.FloatTensor: [B] the path score
        """
        if self.batch_first:
            unary = unary.transpose(0, 1)
        trans = self.transitions  # [1, N, N]
        path, score = viterbi(unary, trans, lengths, self.start_idx, self.end_idx)
        if self.batch_first:
            path = path.transpose(0, 1)
        return path, score


class SequenceModel(nn.Module):

    def __init__(self, nc, embeddings, transducer, decoder=None):
        super().__init__()
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
        decoder_model = CRF(nc, batch_first=False) if decoder is None else decoder
        super().__init__(nc, embeddings, transducer, decoder_model)
        self.path_scores = None

    def neg_log_loss(self, unary, tags, lengths):
        return self.decoder_model.neg_log_loss(unary, tags, lengths)

    def forward(self, inputs):
        time_first, self.path_scores = super().forward(inputs)
        return time_first.transpose(0, 1)


class LangSequenceModel(nn.Module):

    def __init__(self, nc, embeddings, transducer, decoder=None, name=None):
        super().__init__()
        if isinstance(embeddings, dict):
            self.embed_model = EmbeddingsStack(embeddings)
        else:
            assert isinstance(embeddings, EmbeddingsStack)
            self.embed_model = embeddings
        self.transducer_model = transducer
        if hasattr(transducer, 'requires_state') and transducer.requires_state:
            self._call = self._call_with_state
            self.requires_state = True
        else:
            self._call = self._call_without_state
            self.requires_state = False
        self.output_layer = nn.Linear(self.transducer_model.output_dim, nc)
        self.decoder_model = decoder

    def forward(self, inputs):
        return self._call(inputs)

    def _call_with_state(self, inputs):

        h = inputs.get('h')

        embedded = self.embed_model(inputs)
        transduced, hidden = self.transducer_model((embedded, h))
        transduced = self.output_layer(transduced)
        return transduced, hidden

    def _call_without_state(self, inputs):
        embedded = self.embed_model(inputs)
        transduced = self.transducer_model((embedded))
        transduced = self.output_layer(transduced)
        return transduced, None


def pytorch_embedding(weights, finetune=True):
    lut = nn.Embedding(weights.shape[0], weights.shape[1], padding_idx=0)
    del lut.weight
    lut.weight = nn.Parameter(torch.FloatTensor(weights),
                              requires_grad=finetune)
    return lut


def subsequent_mask(size):
    """
    Creates a lower triangular mask to mask future

    :param size: Temporal length
    :return: A tensor of type `uint8` that is 1s along diagonals and below, zero  o.w
    """
    attn_shape = (1, 1, size, size)
    sub_mask = np.tril(np.ones(attn_shape)).astype('uint8')
    return torch.from_numpy(sub_mask)


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention from https://arxiv.org/abs/1706.03762 via http://nlp.seas.harvard.edu/2018/04/03/attention.html

    Multi-headed attention provides multiple looks of low-order projections K, Q and V using an attention function
    (specifically `scaled_dot_product_attention` in the paper.  This allows multiple relationships to be illuminated
    via attention on different positional and representational information from each head.

    The number of heads `h` times the low-order projection dim `d_k` is equal to `d_model` (which is asserted upfront).
    This means that each weight matrix can be simply represented as a linear transformation from `d_model` to `d_model`,
    and partitioned into heads after the fact.

    Finally, an output projection is applied which brings the output space back to `d_model`, in preparation for the
    sub-sequent `FFN` sub-layer.

    There are 3 uses of multi-head attention in the Transformer.
    For encoder-decoder layers, the queries come from the previous decoder layer, and the memory keys come from
    the encoder.  For encoder layers, the K, Q and V all come from the output of the previous layer of the encoder.
    And for self-attention in the decoder, K, Q and V all come from the decoder, but here it is masked to prevent using
    future values
    """
    def __init__(self, num_heads, d_model, dropout=0.1, scale=False):
        """Constructor for multi-headed attention

        :param h: The number of heads
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param attn_fn: A function to apply attention, defaults to SDP
        """
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.h = num_heads
        self.w_Q = Dense(d_model, d_model)
        self.w_K = Dense(d_model, d_model)
        self.w_V = Dense(d_model, d_model)
        self.w_O = Dense(d_model, d_model)
        self.attn_fn = self._scaled_dot_product_attention if scale else self._dot_product_attention
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def _scaled_dot_product_attention(self, query, key, value, mask=None, dropout=None):
        """Scaled dot product attention, as defined in https://arxiv.org/abs/1706.03762

        We apply the query to the keys to recieve our weights via softmax, which are then applied
        for each value, but in a series of efficient matrix operations.  In the case of self-attention,
        the key, query and values are all low order projections of the same input.

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param value: a set of values from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :param dropout: apply dropout operator post-attention (this is not a float)
        :return: A tensor that is (BxHxTxT)

        """
        # (., H, T, T) = (., H, T, D) x (., H, D, T)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        if dropout is not None:
            weights = dropout(weights)
        return torch.matmul(weights, value), weights

    def _dot_product_attention(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, qkvm):
        """Low-order projections of query, key and value into multiple heads, then attention application and dropout

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param value: a set of values from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: Multi-head attention output, result of attention application to sequence (B, T, d_model)
        """
        query, key, value, mask = qkvm
        batchsz = query.size(0)

        # (B, H, T, D)
        query = self.w_Q(query).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_K(key).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_V(value).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = self.attn_fn(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(batchsz, -1, self.h * self.d_k)
        return self.w_O(x)


class TransformerEncoder(nn.Module):
    def __init__(self, num_heads, d_model, pdrop, scale=True, activation_type='relu', d_ff=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, scale=scale)
        self.ffn = nn.Sequential(Dense(self.d_model, self.d_ff),
                                 get_activation(activation_type),
                                 Dense(self.d_ff, self.d_model))
        self.ln1 = nn.LayerNorm(self.d_model, eps=1e-6)
        self.ln2 = nn.LayerNorm(self.d_model, eps=1e-6)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, inputs):
        """
        :param inputs: `(x, mask)`
        :return: The output tensor
        """
        x, mask = inputs

        x = self.ln1(x)
        h = self.self_attn((x, x, x, mask))
        x = x + self.dropout(h)

        x = self.ln2(x)
        x = x + self.dropout(self.ffn(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_heads, d_model, pdrop, scale=True, activation_type='relu', d_ff=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.self_attn = MultiHeadedAttention(num_heads, self.d_model, pdrop, scale=scale)
        self.src_attn = MultiHeadedAttention(num_heads, self.d_model, pdrop, scale=scale)
        self.ffn = nn.Sequential(Dense(self.d_model, self.d_ff),
                                 get_activation(activation_type),
                                 Dense(self.d_ff, self.d_model))

        self.ln1 = nn.LayerNorm(self.d_model, eps=1e-6)
        self.ln2 = nn.LayerNorm(self.d_model, eps=1e-6)
        self.ln3 = nn.LayerNorm(self.d_model, eps=1e-6)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, inputs):

        x, memory, src_mask, tgt_mask = inputs
        x = self.ln1(x)
        x = x + self.dropout(self.self_attn((x, x, x, tgt_mask)))

        x = self.ln2(x)
        x = x + self.dropout(self.src_attn((x, memory, memory, src_mask)))

        x = self.ln3(x)
        x = x + self.dropout(self.ffn(x))
        return x


class TransformerEncoderStack(nn.Module):

    def __init__(self, num_heads, d_model, pdrop, scale=True, layers=1, activation='relu', d_ff=None, **kwargs):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.ln = nn.LayerNorm(d_model, eps=1e-6)
        for i in range(layers):
            self.encoders.append(TransformerEncoder(num_heads, d_model, pdrop, scale, activation, d_ff))

    def forward(self, inputs):
        x, mask = inputs
        for layer in self.encoders:
            x = layer((x, mask))
        return self.ln(x)


class TransformerDecoderStack(nn.Module):
    def __init__(self, num_heads, d_model, pdrop, scale=True, layers=1, activation_type='relu', d_ff=None):
        super().__init__()
        self.decoders = nn.ModuleList()
        self.ln = nn.LayerNorm(d_model, eps=1e-6)
        for i in range(layers):
            self.decoders.append(TransformerDecoder(num_heads, d_model, pdrop, scale, activation_type, d_ff))

    def forward(self, inputs):
        x, memory, src_mask, tgt_mask = inputs
        for layer in self.decoders:
            x = layer((x, memory, src_mask, tgt_mask))
        return self.ln(x)


def update_lengths(lengths, eoses, idx):
    """Update the length of a generated tensor based on the first EOS found.

    This is useful for a decoding situation where tokens after an EOS
    can be something other than EOS. This also makes sure that a second
    generated EOS doesn't affect the lengths.

    :param lengths: `torch.LongTensor`: The lengths where zero means an
        unfinished sequence.
    :param eoses:  `torch.ByteTensor`: A mask that has 1 for sequences that
        generated an EOS.
    :param idx: `int`: What value to fill the finished lengths with (normally
        the current decoding timestep).

    :returns: `torch.Tensor`: The updated lengths tensor (same shape and type).
    """
    # If a length is 0 it has never had a length set so it is eligible to have
    # this EOS be the length.
    updatable_lengths = (lengths == 0)
    # If this length can be updated AND this token is an eos
    lengths_mask = updatable_lengths & eoses
    return lengths.masked_fill(lengths_mask, idx)


def gnmt_length_penalty(lengths, alpha=0.8):
    """Calculate a length penalty from https://arxiv.org/pdf/1609.08144.pdf

    The paper states the penalty as (5 + |Y|)^a / (5 + 1)^a. This is implemented
    as ((5 + |Y|) / 6)^a for a (very) tiny performance boost

    :param lengths: `torch.LongTensor`: [B, K] The lengths of the beams.
    :param alpha: `float`: A hyperparameter. See Table 2 for a search on this
        parameter.

    :returns:
        `torch.FloatTensor`: [B, K, 1] The penalties.
    """
    lengths = lengths.to(torch.float)
    penalty = torch.pow(((5 + lengths) / 6), alpha)
    return penalty.unsqueeze(-1)


def no_length_penalty(lengths):
    """A dummy function that returns a no penalty (1)."""
    return torch.ones_like(lengths).to(torch.float).unsqueeze(-1)


def repeat_batch(t, K, dim=0):
    """Repeat a tensor while keeping the concept of a batch.

    :param t: `torch.Tensor`: The tensor to repeat.
    :param K: `int`: The number of times to repeat the tensor.
    :param dim: `int`: The dimension to repeat in. This should be the
        batch dimension.

    :returns: `torch.Tensor`: The repeated tensor. The new shape will be
        batch size * K at dim, the rest of the shapes will be the same.

    Example::

        >>> a = torch.arange(10).view(2, -1)
        >>> a
	tensor([[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9]])
	>>> a.repeat(2, 1)
	tensor([[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9],
		[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9]])
	>>> repeat_batch(a, 2)
	tensor([[0, 1, 2, 3, 4],
		[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9],
		[5, 6, 7, 8, 9]])
    """
    shape = t.shape
    tiling = [1] * (len(shape) + 1)
    tiling[dim + 1] = K
    tiled = t.unsqueeze(dim + 1).repeat(tiling)
    old_bsz = shape[dim]
    new_bsz = old_bsz * K
    new_shape = list(shape[:dim]) + [new_bsz] + list(shape[dim+1:])
    return tiled.view(new_shape)


class BeamSearchBase:

    def __init__(self, beam=1, length_penalty=None, **kwargs):
        self.length_penalty = length_penalty if length_penalty else no_length_penalty
        self.K = beam

    def init(self, encoder_outputs):
        pass

    def step(self, paths, extra):
        pass

    def update(self, beams, extra):
        pass

    def __call__(self, encoder_outputs, **kwargs):
        """Perform batched Beam Search.

        Note:
            The paths and lengths generated do not include the <GO> token.

        :param encoder_outputs: `namedtuple` The outputs of the encoder class.
        :param init: `Callable(ecnoder_outputs: encoder_outputs, K: int)` -> Any: A
            callable that is called once at the start of the search to initialize
            things. This returns a blob that is passed to other callables.
        :param step: `Callable(paths: torch.LongTensor, extra) -> (probs: torch.FloatTensor, extra):
            A callable that is does a single decoding step. It returns the log
            probabilities over the vocabulary in the last dimension. It also returns
            any state the decoding process needs.
        :param update: `Callable(beams: torch.LongTensor, extra) -> extra:
            A callable that is called to edit the decoding state based on the selected
            best beams.
        :param length_penalty: `Callable(lengths: torch.LongTensor) -> torch.floatTensor
            A callable that generates a penalty based on the lengths. Lengths is
            [B, K] and the returned penalty should be [B, K, 1] (or [B, K, V] to
            have token based penalties?)

        :Keyword Arguments:
        * *beam* -- `int`: The number of beams to use.
        * *mxlen* -- `int`: The max number of steps to run the search for.

        :returns:
            tuple(preds: torch.LongTensor, lengths: torch.LongTensor, scores: torch.FloatTensor)
            preds: The predicted values: [B, K, max(lengths)]
            lengths: The length of each prediction [B, K]
            scores: The score of each path [B, K]
        """
        mxlen = kwargs.get('mxlen', 100)
        bsz = encoder_outputs.output.shape[0]
        device = encoder_outputs.output.device
        with torch.no_grad():
            extra = self.init(encoder_outputs)
            paths = torch.full((bsz, self.K, 1), Offsets.GO, dtype=torch.long, device=device)
            # This tracks the log prob of each beam. This is distinct from score which
            # is based on the log prob and penalties.
            log_probs = torch.zeros((bsz, self.K), dtype=torch.float, device=device)
            # Tracks the lengths of the beams, unfinished beams have lengths of zero.
            lengths = torch.zeros((bsz, self.K), dtype=torch.long, device=device)

            for i in range(mxlen - 1):
                probs, extra = self.step(paths, extra)
                V = probs.shape[-1]
                probs = probs.view((bsz, self.K, V))  # [B, K, V]
                if i > 0:
                    # This mask is for all beams that are done.
                    done_mask = (lengths != 0).unsqueeze(-1)  # [B, K, 1]
                    # Can creating this mask be moved out of the loop? It never changes but we don't have V
                    # This mask selects the EOS token
                    eos_mask = torch.zeros((1, 1, V), dtype=done_mask.dtype, device=device)
                    eos_mask[:, :, Offsets.EOS] = 1
                    # This mask selects the EOS token of only the beams that are done.
                    mask = done_mask & eos_mask
                    # Put all probability mass on the EOS token for finished beams.
                    # Otherwise as the other beams get longer they will all give
                    # up and eventually select this beam and all outputs become
                    # the same.
                    probs = probs.masked_fill(done_mask, -np.inf)
                    probs = probs.masked_fill(mask, 0)
                    probs = log_probs.unsqueeze(-1) + probs  # [B, K, V]
                    # Calculate the score of the beam based on the current length.
                    path_scores = probs / self.length_penalty(lengths.masked_fill(lengths == 0, i+1))
                else:
                    # On the first step we only look at probabilities for the first beam.
                    # If we don't then the probs will be the same for each beam
                    # This means the same token will be selected for each beam
                    # And we won't get any diversity.
                    # Using only the first beam ensures K different starting points.
                    path_scores = probs[:, 0, :]

                flat_scores = path_scores.view(bsz, -1)  # [B, K * V]
                best_scores, best_idx = flat_scores.topk(self.K, 1)
                # Get the log_probs of the best scoring beams
                log_probs = probs.view(bsz, -1).gather(1, best_idx).view(bsz, self.K)

                best_beams = best_idx / V  # Get which beam it came from
                best_idx = best_idx % V  # Get the index of the word regardless of which beam it is.

                # Best Beam index is relative within the batch (only [0, K)).
                # This makes the index global (e.g. best beams for the second
                # batch example is in [K, 2*K)).
                offsets = torch.arange(bsz, dtype=torch.long, device=device) * self.K
                offset_beams = best_beams + offsets.unsqueeze(-1)
                flat_beams = offset_beams.view(bsz * self.K)
                # Select the paths to extend based on the best beams
                flat_paths = paths.view(bsz * self.K, -1)
                new_paths = flat_paths[flat_beams, :].view(bsz, self.K, -1)
                # Add the selected outputs to the paths
                paths = torch.cat([new_paths, best_idx.unsqueeze(-1)], dim=2)

                # Select the lengths to keep tracking based on the valid beams left.
                lengths = lengths.view(-1)[flat_beams].view((bsz, self.K))

                extra = self.update(flat_beams, extra)

                # Updated lengths based on if we hit EOS
                last = paths[:, :, -1]
                eoses = (last == Offsets.EOS)
                lengths = update_lengths(lengths, eoses, i + 1)
                if (lengths != 0).all():
                    break
            else:
                # This runs if the loop didn't break meaning one beam hit the max len
                # Add an EOS to anything that hasn't hit the end. This makes the scores real.
                probs, extra = self.step(paths, extra)

                V = probs.size(-1)
                probs = probs.view((bsz, self.K, V))
                probs = probs[:, :, Offsets.EOS]  # Select the score of EOS
                # If any of the beams are done mask out the score of this EOS (they already had an EOS)
                probs = probs.masked_fill((lengths != 0), 0)
                log_probs = log_probs + probs
                end_tokens = torch.full((bsz, self.K, 1), Offsets.EOS, device=device, dtype=paths.dtype)
                paths = torch.cat([paths, end_tokens], dim=2)
                lengths = update_lengths(lengths, torch.ones_like(lengths) == 1, mxlen)
                best_scores = log_probs / self.length_penalty(lengths).squeeze(-1)

        # Slice off the Offsets.GO token
        paths = paths[:, :, 1:]
        return paths, lengths, best_scores

import torch
import numpy as np
from baseline.utils import lookup_sentence, get_version
from baseline.utils import crf_mask as crf_m
from torch.autograd import Variable
import torch.autograd
import torch.nn as nn
import torch.nn.functional
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
        rnn = torch.nn.LSTM(insz, hsz, nlayers, dropout=dropout, bidirectional=True)
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
        nn.init.xavier_uniform_(rnn.weight_hh_l0)
        nn.init.xavier_uniform_(rnn.weight_ih_l0)

    return rnn, ndir*hsz


class LSTMEncoder(nn.Module):

    def __init__(self, insz, hsz, rnntype, nlayers, dropout, residual=False, unif=0, initializer=None):
        super(LSTMEncoder, self).__init__()
        self.residual = residual
        self.rnn, self.outsz = pytorch_lstm(insz, hsz, rnntype, nlayers, dropout, unif, False, initializer)

    def forward(self, tbc, lengths):

        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths.tolist())
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        return output + tbc if self.residual else output


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


#def beam_decode_tokens(model, src_tokens, K, idx2word, mxlen=50):
#    src = prepare_src(model, src_tokens, mxlen)
#    paths, scores = beam_decode(model, src, K)
#    path_str = []
#    for j, path in enumerate(paths):
#        path_str.append([idx2word[i] for i in path])
#    return path_str, scores
    #return beam_decode(model, src, K)


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
    
    for src_len_i, src_i, tgt_i in zip(src_len, src_array, tgt_array):

        print('========================================================================')
        src_len_i = torch.ones(1).fill_(src_len_i).type_as(src_len)

        sent = lookup_sentence(rlut1, src_i.cpu().numpy(), reverse=reverse)
        print('[OP] %s' % sent)
        sent = lookup_sentence(rlut2, tgt_i.cpu().numpy())
        print('[Actual] %s' % sent)
        src_dict = {'src': torch.autograd.Variable(src_i.view(1, -1), requires_grad=False),
                    'src_len': torch.autograd.Variable(src_len_i, requires_grad=False)}
        dst_i = model.run(src_dict)[0][0]
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


def crf_mask(vocab, span_type, s_idx, e_idx, pad_idx=None):
    """Create a CRF mask.

    Returns a Tensor with valid transitions as a 0 and invalid as a 1 for easy use with `masked_fill`
    """
    np_mask = crf_m(vocab, span_type, s_idx, e_idx, pad_idx=pad_idx)
    return torch.from_numpy(np_mask) == 0


class CRF(nn.Module):
    def __init__(self, n_tags, idxs=None, batch_first=True, vocab=None, span_type=None, pad_idx=None):
        """Initialize the object.
        :param n_tags: int The number of tags in your output (emission size)
        :param idxs: Tuple(int. int) The index of the start and stop symbol
            in emissions.
        :param vocab: The label vocab of the form vocab[string]: int
        :param span_type: The tagging span_type used. `IOB`, `IOB2`, or `IOBES`
        :param pds_idx: The index of the pad symbol in the vocab
        Note:
            if idxs is none then the CRF adds these symbols to the emission
            vectors and n_tags is assumed to be the number of output tags.
            if idxs is not none then the first element is assumed to be the
            start index and the second idx is assumed to be the end index. In
            this case n_tags is assumed to include the start and end symbols.
            if vocab is not None then a transition mask will be created that
            limits illegal transitions.
        """
        super(CRF, self).__init__()

        if idxs is None:
            self.start_idx = n_tags
            self.end_idx = n_tags + 1
            self.n_tags = n_tags + 2
            self.add_ends = True
        else:
            self.start_idx, self.end_idx = idxs
            self.n_tags = n_tags
            self.add_ends = False
        self.span_type = None
        if vocab is not None:
            assert span_type is not None, "To mask transitions you need to provide a tagging span_type, choices are `IOB`, `BIO` (or `IOB2`), and `IOBES`"
            # If there weren't start and end idx provided we need to add them.
            if idxs is None:
                vocab = vocab.copy()
                vocab['<GO>'] = self.start_idx
                vocab['<EOS>'] = self.end_idx
            self.span_type = span_type
            self.register_buffer('mask', crf_mask(vocab, span_type, self.start_idx, self.end_idx, pad_idx).unsqueeze(0))
        else:
            self.mask = None

        self.transitions_p = nn.Parameter(torch.Tensor(1, self.n_tags, self.n_tags).zero_())
        self.batch_first = batch_first

    def extra_repr(self):
        str_ = "n_tags=%d, batch_first=%s" % (self.n_tags, self.batch_first)
        if self.mask is not None:
            str_ += ", masked=True, span_type=%s" % self.span_type
        return str_

    @staticmethod
    def _prep_input(input_):
        ends = torch.Tensor(input_.size()[0], input_.size()[1], 2).fill_(-1e4).to(input_.device)
        return torch.cat([input_, ends], dim=2)

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
        if self.add_ends:
            unary = CRF._prep_input(unary)
        _, batch_size, _ = unary.size()
        min_lengths = torch.min(lengths)
        fwd_score = self.forward(unary, lengths, batch_size, min_lengths)
        gold_score = self.score_sentence(unary, tags, lengths, batch_size, min_lengths)
        return fwd_score - gold_score

    def score_sentence(self, unary, tags, lengths, batch_size, min_length):
        """Score a batch of sentences.

        :param unary: torch.FloatTensor: [T, B, N]
        :param tags: torch.LongTensor: [T, B]
        :param lengths: torch.LongTensor: [B]
        :param batzh_size: int: B
        :param min_length: torch.LongTensor: []

        :return: torch.FloatTensor: [B]
        """
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

    def forward(self, unary, lengths, batch_size, min_length):
        """For CRF forward on a batch.

        :param unary: torch.FloatTensor: [T, B, N]
        :param lengths: torch.LongTensor: [B]
        :param batzh_size: int: B
        :param min_length: torch.LongTensor: []

        :return: torch.FloatTensor: [B]
        """
        # alphas: [B, 1, N]
        alphas = torch.Tensor(batch_size, 1, self.n_tags).fill_(-1e4).to(unary.device)
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

    def decode(self, unary, lengths):
        """Do Viterbi decode on a batch.

        :param unary: torch.FloatTensor: [T, B, N] or [B, T, N]
        :param lengths: torch.LongTensor: [B]

        :return: List[torch.LongTensor]: [B] the paths
        :return: torch.FloatTensor: [B] the path score
        """
        if self.batch_first:
            unary = unary.transpose(0, 1)
        if self.add_ends:
            unary = CRF._prep_input(unary)
        seq_len, batch_size, _ = unary.size()
        min_length = torch.min(lengths)
        batch_range = torch.arange(batch_size, dtype=torch.int64)
        backpointers = []

        # alphas: [B, 1, N]
        alphas = torch.Tensor(batch_size, 1, self.n_tags).fill_(-1e4).to(unary.device)
        alphas[:, 0, self.start_idx] = 0
        alphas.requires_grad = True

        trans = self.transitions  # [1, N, N]

        for i, unary_t in enumerate(unary):
            # Broadcast alphas along the rows of trans and trans along the batch of alphas
            next_tag_var = alphas + trans  # [B, 1, N] + [1, N, N] -> [B, N, N]
            viterbi, best_tag_ids = torch.max(next_tag_var, 2) # [B, N]
            backpointers.append(best_tag_ids.data)
            new_alphas = viterbi + unary_t  # [B, N] + [B, N]
            new_alphas.unsqueeze_(1)  # Prep for next round
            # If we haven't reached your length zero out old alpha and take new one.
            # If we are past your length, zero out new_alpha and keep old one.
            if i >= min_length:
                mask = (i < lengths).view(-1, 1, 1)
                alphas = alphas.masked_fill(mask, 0) + new_alphas.masked_fill(mask == 0, 0)
            else:
                alphas = new_alphas

        # Add end tag
        terminal_var = alphas.squeeze(1) + trans[:, self.end_idx]
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

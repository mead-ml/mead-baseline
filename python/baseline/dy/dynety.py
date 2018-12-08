import math
from itertools import chain
import numpy as np
import dynet as dy
from baseline.utils import (
    lookup_sentence,
    transition_mask as transition_mask_np,
    sequence_mask as seq_mask
)


def transition_mask(vocab, span_type, s_idx, e_idx, pad_idx):
    mask = transition_mask_np(vocab, span_type, s_idx, pad_idx)
    inv_mask = (mask == 0).astype(np.float32)
    return mask, inv_mask


class DynetModel(object):
    def __init__(self, pc=None):
        super(DynetModel, self).__init__()
        self._pc = pc
        self.train = True

    @property
    def pc(self):
        return self._pc

    @pc.setter
    def pc(self, value):
        self._pc = value

    def __str__(self):
        str_ = []
        for p in chain(self.pc.lookup_parameters_list(), self.pc.parameters_list()):
            str_.append("{}: {}".format(p.name(), p.shape()))
        return '\n'.join(str_)


class DynetLayer(DynetModel): pass


def sequence_mask(lengths, max_len=-1):
    """Build a sequence mask for dynet.

    This is a bit weird, most of the time we have dynet as H, T so it would
    seem like we would want the mask to be ((1, T), B) but the only places
    where we do the masking right now is in attention and it makes sense to
    have it shaped as ((T, 1), B).
    """
    mask = seq_mask(lengths, max_len)
    mask = np.expand_dims(np.transpose(mask), 1)
    inv_mask = (mask == 0).astype(np.uint8)
    return dy.inputTensor(mask, batched=True), dy.inputTensor(inv_mask, batched=True)


def unsqueeze(x, dim):
    """Add a dimension of size 1 to `x` at position `dim`."""
    shape, batchsz = x.dim()
    dim = len(shape) + dim + 1 if dim < 0 else dim
    shape = list(shape)
    shape.insert(dim, 1)
    return dy.reshape(x, tuple(shape), batch_size=batchsz)


def squeeze(x, d=-1):
    shape, batchsz = x.dim()
    if d == -1:
        shape = tuple(filter(lambda x: x != 1, shape))
    else:
        assert shape[d] == 1, "Cannot squeeze dimension {} of size {}".format(d, shape[d])
        shape = list(shape)
        _ = shape.pop(d)
        shape = tuple(shape)
    return dy.reshape(x, shape, batch_size=batchsz)


def batch_matmul(x, y):
    """Matmul between first two layers but the rest are ignored.

    Input: ((X, Y, ..), B) and ((Y, Z, ..), B)
    Output: ((X, Z, ..), B)
    """
    x_shape, batchsz = x.dim()
    x_mat = x_shape[:2]
    sames = x_shape[2:]
    fold = np.prod(sames)
    y_shape, _ = y.dim()
    y_mat = y_shape[:2]

    x = dy.reshape(x, x_mat, batch_size=fold*batchsz)
    y = dy.reshape(y, y_mat, batch_size=fold*batchsz)

    z = x * y
    z = dy.reshape(z, tuple([x_mat[0], y_mat[1]] + list(sames)), batch_size=batchsz)
    return z


def folded_softmax(x, softmax=dy.softmax):
    """Dynet only allows for softmax on matrices."""
    shape, batchsz = x.dim()
    first = shape[0]
    flat = np.prod(shape[1:])
    x = dy.reshape(x, (first, flat), batch_size=batchsz)
    x = softmax(x, d=0)
    return dy.reshape(x, shape, batch_size=batchsz)


def transpose(x, dim1, dim2):
    """Swap dimensions `dim1` and `dim2`."""
    shape, _ = x.dim()
    dims = list(range(len(shape)))
    tmp = dims[dim1]
    dims[dim1] = dims[dim2]
    dims[dim2] = tmp
    return dy.transpose(x, dims=dims)


def dynet_activation(type_):
    """Get activation functions based on names."""
    return dy.rectify


def LayerNorm(num_features, pc, name='layer-norm'):
    pc = pc.add_subcollection(name=name)
    a = pc.add_parameters(num_features, name='a')
    b = pc.add_parameters(num_features, name='b')

    def norm(x):
        """Layer Norm only handles a vector in dynet so fold extra dims into the batch."""
        shape, batchsz = x.dim()
        first = shape[0]
        fold = np.prod(shape[1:])
        x = dy.reshape(x, (first,), batch_size=batchsz*fold)
        x = dy.layer_norm(x, a, b)
        return dy.reshape(x, shape, batch_size=batchsz)

    return norm


class Linear(DynetLayer):
    def __init__(self, osz, isz, pc, name="linear"):
        """
        :param osz: int
        :param isz: int
        :param pc: dy.ParameterCollection
        :param name: str
        """
        pc = pc.add_subcollection(name=name)
        super(Linear, self).__init__(pc)
        self.weight = self.pc.add_parameters((osz, isz), name="weight")
        self.bias = self.pc.add_parameters((osz,), name="bias")

    def __call__(self, input_, train=False):
        """
        :param input_: dy.Expression ((isz,), B)

        Returns:
            dy.Expression ((osz,), B)
        """
        # Affine Transformation squeezes out a final dim of 1 breaking it when
        # This func if used for ((H, T), B) -> ((O, T), B)
        # return dy.affine_transform([bias, weight, input_])
        return self.bias + self.weight * input_


def squeeze_and_transpose(x):
    return dy.transpose(squeeze(x))


class WeightShareLinear(DynetLayer):
    default = 'weight-shared'
    def __init__(self, osz, weight, pc, transform=None, name=None):
        name = self.default if name is None else self.default + self.clean(name)
        pc = pc.add_subcollection(name=name)
        super(WeightShareLinear, self).__init__(pc)
        self.weight = weight
        self.bias = self.pc.add_parameters((osz,), name='bias')
        self.transform = transform if transform is not None else lambda x: x

    def __call__(self, input_):
        a = self.transform(self.weight)
        return self.bias + self.transform(self.weight) * input_

    @staticmethod
    def clean(name):
        return name.replace('/', '-').replace('_', '-')[:-1]


def HighwayConnection(funcs, sz, pc, name="highway"):
    """Highway Connection around arbitrary functions.

    This highway block creates highway connections that short circuit each function in funcs.

    :param funcs: A list of functions you can pass input_ to
    :param sz: int The size of the input
    :param pc: dy.ParameterCollection
    :name str: The name of the layer
    """
    highway_pc = pc.add_subcollection(name=name)
    weights = []
    biases = []
    for i in range(len(funcs)):
        weights.append(highway_pc.add_parameters((sz, sz), name="weight-{}".format(i)))
        biases.append(highway_pc.add_parameters((sz), init=dy.ConstInitializer(-2), name="bias-{}".format(i)))

    def highway(input_, train):
        for func, weight, bias in zip(funcs, weights, biases):
            proj = dy.rectify(func(input_, train))
            transform = dy.logistic(dy.affine_transform([bias, weight, input_]))
            input_ = dy.cmult(transform, proj) + dy.cmult(input_, 1 - transform)
        return input_

    return highway


def SkipConnection(funcs, *args, **kwargs):
    def skip(input_, train):
        for func in funcs:
            proj = func(input_, train)
            input_ = input_ + proj
        return input_
    return skip


# RNN functions
def rnn_forward(rnn, input_):
    """Return only the output of the final layer.

    :param rnn: dy.RNNBuild or dy.BiRNNBuilder
    :param input_: List[dy.Expression]

    Returns:
        List[dy.Expression]: The outputs
    """
    if isinstance(rnn, dy.BiRNNBuilder):
        return rnn.transduce(input_)
    state = rnn.initial_state()
    return state.transduce(input_)


def rnn_forward_with_state(rnn, input_, lengths=None, state=None, batched=True, backward=False):
    """Return the output of the final layers and the final state of the RNN.

    :param rnn: dy.RNNBuilder
    :param input_: List[dy.Expression]
    :param lengths: List[int]
    :param state: List[np.ndarray] The previous state (used in TBPTT)
    :param batched: bool Is the state batched?
    :param backward: bool Is this a backward rnn in a bRNN?

    Returns:
        List[dy.Expression] (Seq_len): The outputs
        List[dy.Expression] (2 * layers if lstm): The state
    """
    if state is not None:
        state = [dy.inputTensor(s, batched) for s in state]
    lstm_state = rnn.initial_state(state)
    if backward:
        states = lstm_state.add_inputs(reversed(input_))
        outputs = list(reversed([s.h()[-1] for s in states]))
        # When going backwards (we pad right) the final state of the rnn
        # is always the last one.
        final_state = states[-1].s()
        return outputs, final_state
    states = lstm_state.add_inputs(input_)
    outputs = [s.h()[-1] for s in states]
    if lengths is None:
        if backward:
            outputs = list(reversed(outputs))
        return outputs, states[-1].s()
    final_states = [states[l - 1].s() for l in lengths]
    final_state_by_batch = []
    for i, state in enumerate(final_states):
        batch_state = [dy.pick_batch_elem(s, i) for s in state]
        final_state_by_batch.append(batch_state)
    final_state = []
    for i in range(len(final_state_by_batch[0])):
        col = dy.concatenate_to_batch([final_state_by_batch[j][i] for j in range(len(final_state_by_batch))])
        final_state.append(col)
    if backward:
        outputs = list(reversed(outputs))
    return outputs, final_state


def rnn_encode(rnn, input_, lengths):
    """Return the final output for each batch based on lengths.

    :param rnn: dy.RNNBuilder or dy.BiRNNBuilder
    :param input_: List[dy.Expression]
    :param lengths: List[int]

    Returns:
        dy.Expression
    """
    states = rnn_forward(rnn, input_)
    final_states = [dy.pick_batch_elem(states[l - 1], i) for i, l in enumerate(lengths)]
    return dy.concatenate_to_batch(final_states)


def Convolution1d(fsz, cmotsz, dsz, pc, strides=(1, 1, 1, 1), activation_type="relu", name="conv"):
    """1D Convolution.

    :param fsz: int, Size of conv filter.
    :param cmotsz: int, Size of conv output.
    :param dsz: int, Size of the input.
    :param pc: dy.ParameterCollection
    :param strides: Tuple[int, int, int, int]
    """
    conv_pc = pc.add_subcollection(name=name)
    fan_in = dsz * fsz
    fan_out = cmotsz * fsz
    # Pytorch and Dynet have a gain param that has suggested values based on
    # the nonlinearity type, this defaults to the one for relu atm.
    glorot_bounds = 0.5 * np.sqrt(6.0 / (fan_in + fan_out))
    weight = conv_pc.add_parameters(
        (1, fsz, dsz, cmotsz),
        init=dy.UniformInitializer(glorot_bounds),
        name='weight'
    )
    bias = conv_pc.add_parameters((cmotsz), name="bias")
    act = dynet_activation(activation_type)

    def conv(input_, _=None):
        """Perform the 1D conv.

        :param input: dy.Expression ((1, T, dsz), B)

        Returns:
            dy.Expression ((cmotsz,), B)
        """
        c = dy.conv2d_bias(input_, weight, bias, strides, is_valid=False)
        return act(c)

    return conv


def mot_pool(x, strides=(1, 1, 1, 1)):
    # dy.max_dim(x, d=0) is currently slow (see https://github.com/clab/dynet/issues/1011)
    # So we do the max using max pooling instead.
    ((_, seq_len, cmotsz), _) = x.dim()
    pooled = dy.maxpooling2d(x, [1, seq_len, 1], strides)
    return dy.reshape(pooled, (cmotsz,))


def dynet_activation(name='relu'):
    if name == 'tahn':
        return dy.tahn
    if name == 'sigmoid':
        return dy.logistic
    if name == 'log_sigmoid':
        return dy.log_sigmoid
    return dy.rectify


def ConvEncoder(filtsz, outsz, insz, pdrop, pc, layers=1, activation_type='relu'):
    conv = Convolution1d(filtsz, outsz, insz, pc, activation_type=activation_type)

    def encode(input_, train):
        x = conv(input_)
        x = dy.dropout(x, pdrop) if train else x
        return x

    return encode


def ConvEncoderStack(filtsz, outsz, insz, pdrop, pc, layers=1, activation_type='relu'):
    first_layer = ConvEncoder(filtsz, outsz, insz, pdrop, pc, activation_type=activation_type)
    later_layers = [ConvEncoder(filtsz, outsz, outsz, pdrop, pc, activation_type) for _ in range(layers - 1)]
    residual = SkipConnection(later_layers)

    def encode(input_, train):
        dims = tuple([1] + list(input_.dim()[0]))
        input_ = dy.reshape(input_, dims)
        x = first_layer(input_, train)
        x = residual(x, train)
        new_shape = x.dim()[0]
        x = dy.reshape(x, new_shape[1:])
        return x

    return encode


def ParallelConv(filtsz, cmotsz, dsz, pc, strides=(1, 1, 1, 1), name="parallel-conv"):
    if isinstance(cmotsz, int):
        cmotsz = [cmotsz] * len(filtsz)
    conv_pc = pc.add_subcollection(name=name)
    convs = [Convolution1d(fsz, cmot, dsz, conv_pc, strides, name="conv-{}".format(fsz)) for fsz, cmot in zip(filtsz, cmotsz)]

    def conv(input_, _=None):
        dims = tuple([1] + list(input_.dim()[0]))
        input_ = dy.reshape(input_, dims)
        mots = []
        for conv in convs:
            mots.append(mot_pool(conv(input_)))
        return dy.concatenate(mots)

    return conv


def attention(attn_type='bahdanau', *args, **kwargs):
    attn_type= attn_type.lower()
    if attn_type == 'dot':
        return DotProductAttention(*args, **kwargs)
    if attn_type in {'concat', 'bahdanau'}:
        return BahdanauAttention(*args, **kwargs)
    if attn_type == 'sdp':
        return ScaledDotProductAttention(*args, **kwargs)
    return LuongAttention(*args, **kwargs)


class Attention(DynetLayer):
    """Attention optimized for encoder decoder where the keys and values are the same."""
    def __init__(self, hsz, pc, name):
        pc = pc.add_subcollection(name=name)
        super(Attention, self).__init__(pc)
        self.hsz = hsz
        self.down_proj = Linear(hsz, 2 * hsz, self.pc, name='combine')

    def cache_encoder(self, context_vectors):
        """Cache transformations to the encoder vectors.

        :param context_vectors: list[dy.Expression] The encoder output vectors
            we do attention over. List is of lengths T and expression are ((H,), B)
        """
        self.context = dy.concatenate_cols(context_vectors)  # ((H, T), B)

    def __call__(self, query, mask=None):
        """Do attention over the encoders with query.

        :param query: dy.Expression ((H,), B)
        :param mask: tuple(dy.Expression) The masks for attention. ((T, 1), B)

        :returns: dy.Expression ((H,), B)
        """
        attn = self._attend(query, mask)
        return self._combine(attn, query)

    def _attend(self, query, mask):
        """Calculate the attention weights.

        :param query: dy.Expression ((H,), B)
        :param mask: tuple(dy.Expression) The masks for attention. ((T, 1), B)

        :returns: dy.Expression ((H,), B)
        """
        pass


    def _combine(self, attn, query):
        """Combine the encoder vectors weighted by the attention weights.

        :param attn: dy.Expression ((H,), B)
        :param query: dy.Expression ((H,), B)

        :return: dy.Expression ((H,), B)
        """
        # self.context ((H, T), B)
        # attn         ((T, 1), B)
        # Encoder Vectors has data along the columns so a matmul with the
        # weights (which are also a column) creates a sum of encoder weighted
        # by attention weights
        output_vector = self.context * attn  # ((H, 1), B)
        output_vector = squeeze(output_vector, 1)  # ((H,), B)
        cat = dy.concatenate([output_vector, query])  # ((2 * H,), B)
        return self.down_proj(cat)  # ((H,), B)


class BahdanauAttention(Attention):
    def __init__(self, hsz, pc, name="bahdanau-attention"):
        super(BahdanauAttention, self).__init__(hsz, pc, name)
        self.encoder = self.pc.add_parameters((hsz, hsz), name='encoder-projection')
        self.decoder = self.pc.add_parameters((hsz, hsz), name='decoder-projection')
        self.v = self.pc.add_parameters((1, hsz), name='attention-vector')

    def cache_encoder(self, context_vectors):
        """Cache transformations to the encoder vectors.

        This also projects the context vectors into a new space

        :param context_vectors: list[dy.Expression] The encoder output vectors
            we do attention over. List is of lengths T and expression are ((H,), B)
        """
        self.context = dy.concatenate_cols(context_vectors)  # ((H, T), B)
        self.context_proj = self.encoder * self.context  # ((H, T), B)

    def _attend(self, query, mask=None):
        # query ((H), B)
        # mask  ((T, 1), B)
        projected_state = self.decoder * query  # ((H,), B)
        non_lin = dy.tanh(dy.colwise_add(self.context_proj, projected_state))  # ((H, T), B)
        attn_scores = dy.transpose(self.v * non_lin)  # ((1, H), B) * ((H, T), B) -> ((1, T), B) -> ((T, 1), B)
        if mask is not None:
            attn_scores = dy.cmult(attn_scores, mask[0]) + (mask[1] * dy.scalarInput(-1e9))
        return dy.softmax(attn_scores)  # ((T, 1), B)


class DotProductAttention(Attention):
    def __init__(self, hsz, pc, name="dot-product-attention"):
        super(DotProductAttention, self).__init__(hsz, pc, name)

    def _attend(self, query, mask=None):
        # query ((H), B)
        # mask  ((T, 1), B)
        query = unsqueeze(query, 0)  # ((1, H), B)
        # ((1, H), B) * ((H, T), B) -> ((1, T), B) -> ((T, 1), B)
        attn_scores = dy.transpose(query * self.context)
        if mask is not None:
            attn_scores = dy.cmult(attn_scores, mask[0]) + (mask[1] * dy.scalarInput(-1e9))
        return dy.softmax(attn_scores)  # ((T, 1), B)

    def _combine(self, attn, query):
        comb = super(DotProductAttention, self)._combine(attn, query)  # ((H,), B)
        return dy.tanh(comb)


class ScaledDotProductAttention(Attention):
    def __init__(self, hsz, pc, name="scaled-dot-product-attention"):
        super(ScaledDotProductAttention, self).__init__(hsz, pc, name)
        self.scale = math.sqrt(self.hsz)

    def _attend(self, query, mask=None):
        # query ((H), B)
        # mask  ((T, 1), B)
        query = unsqueeze(query, 0)
        # ((1, H), B) * ((H, T), B) -> ((1, T), B) -> ((T, 1), B)
        attn_scores = dy.cdiv(dy.transpose(query * self.context), dy.scalarInput(self.scale))
        if mask is not None:
            attn_scores = dy.cmult(attn_scores, mask[0]) + (mask[1] * dy.scalarInput(-1e9))
        return dy.softmax(attn_scores)  # ((T, 1), B)

    def _combine(self, attn, query):
        comb = super(ScaledDotProductAttention, self)._combine(attn, query)
        return dy.tanh(comb)


class LuongAttention(Attention):
    def __init__(self, hsz, pc, name="bilinear-attention"):
        super(LuongAttention, self).__init__(hsz, pc, name)
        self.A = self.pc.add_parameters((self.hsz, self.hsz), name="bilinear-weight")

    def cache_encoder(self, context_vectors):
        """Cache the context vectors and project them into a new spaace."""
        self.context = dy.concatenate_cols(context_vectors)  # ((H, T), B)
        self.context_proj = self.A * self.context # ((H, T), B)

    def _attend(self, query, mask=None):
        query = unsqueeze(query, 0) # ((1, H), B)
        # ((1, H), B) * ((H, T), B) -> ((1, T), B) -> ((T, 1), B)
        attn_scores = dy.transpose(query * self.context)
        if mask is not None:
            attn_scores = dy.cmult(attn_scores, mask[0]) + (mask[1] * dy.scalarInput(-1e9))
        return dy.softmax(attn_scores)

    def _combine(self, attn, query):
        comb = super(LuongAttention, self)._combine(attn, query)
        return dy.tanh(comb)


class CRF(DynetModel):
    """Linear Chain CRF in Dynet."""

    def __init__(self, n_tags, pc, idxs=None, mask=None):
        """Initialize the object.

        :param n_tags: int The number of tags in your output (emission size)
        :param pc: dy.ParameterCollection
        :param idxs: Tuple(int. int) The index of the start and stop symbol in emissions.

        Note:
            if idxs is none then the CRF adds these symbols to the emission vectors and
            n_tags is assumed to be the number of output tags.

            if idxs is not none then the first element is assumed to be the start index
            and the second idx is assumed to be the end index. In this case n_tags is
            assumed to include the start and end symbols.

            if vocab is not None then we create a mask to reduce the probability of
            illegal transitions.
        """
        pc = pc.add_subcollection(name="crf")
        super(CRF, self).__init__(pc)
        self.start_idx, self.end_idx = idxs
        self.n_tags = n_tags
        if mask is not None:
            self.mask = mask[0]
            self.inv_mask = mask[1] * -1e4
        else:
            self.mask = mask
        self.transitions_p = self.pc.add_parameters((self.n_tags, self.n_tags), name="transition")

    @property
    def transitions(self):
        if self.mask is not None:
            return dy.cmult(self.transitions_p, dy.inputTensor(self.mask)) + dy.inputTensor(self.inv_mask)
        return self.transitions_p

    def score_sentence(self, emissions, tags):
        """Get the score of a given sentence.

        :param emissions: List[dy.Expression ((H,), B)]
        :param tags: List[int]

        Returns:
            dy.Expression ((1,), B)
        """
        tags = np.concatenate((np.array([self.start_idx], dtype=int), tags))
        score = dy.scalarInput(0)
        transitions = self.transitions
        for i, e in enumerate(emissions):
            # Due to Dynet being column based it is best to use the transition
            # matrix so that x -> y is T[y, x].
            score += dy.pick(dy.pick(transitions, tags[i + 1]), tags[i]) + dy.pick(e, tags[i + 1])

        score += dy.pick(dy.pick(transitions, self.end_idx), tags[-1])
        return score

    def neg_log_loss(self, emissions, tags):
        """Get the Negative Log Loss. T x L

        :param emissions: List[dy.Expression ((H,), B)]
        :param tags: List[int]

        Returns:
            dy.Expression ((1,), B)
        """
        viterbi_score = self._forward(emissions)
        gold_score = self.score_sentence(emissions, tags)
        # CRF Loss: P_real / P_1..N -> -log(CRF Loss)
        # -(log (e^S_real) / (e^S_1..N))
        # -(log (e^S_real) - log(e^S_1..N))
        # -(S_real - S_1..N)
        # S_1..N - S_real
        return viterbi_score - gold_score

    def _forward(self, emissions):

        """Viterbi forward to calculate all path scores.

        :param emissions: List[dy.Expression]

        Returns:
            dy.Expression ((1,), B)
        """
        init_alphas = [-1e4] * self.n_tags
        init_alphas[self.start_idx] = 0

        alphas = dy.inputVector(init_alphas)
        transitions = self.transitions
        # len(emissions) == T
        for emission in emissions:
            add_emission = dy.colwise_add(transitions, emission)
            scores = dy.colwise_add(dy.transpose(add_emission), alphas)
            # dy.logsumexp takes a list of dy.Expression and computes logsumexp
            # elementwise across the lists so for example the logsumexp is calculated
            # for [0] in each list. This means we want the scores for a given
            # transition scores for a tag to be in the columns
            alphas = dy.logsumexp([x for x in scores])
        last_alpha = alphas + dy.pick(transitions, self.end_idx)
        alpha = dy.logsumexp([x for x in last_alpha])
        return alpha

    def decode(self, emissions):
        """Viterbi decode to find the best sequence.

        :param emissions: List[dy.Expression]

        Returns:
            List[int], dy.Expression ((1,), B)
        """
        transition = self.transitions
        return viterbi(emissions, transition, self.start_idx, self.end_idx)


def viterbi(emissions, transition, start_idx, end_idx, norm=False):
    n_tags = emissions[0].dim()[0][0]
    backpointers = []

    inits = [-1e4] * n_tags
    inits[start_idx] = 0
    alphas = dy.inputVector(inits)
    alphas = dy.log_softmax(alphas) if norm else alphas

    for emission in emissions:
        next_vars = dy.colwise_add(dy.transpose(transition), alphas)
        best_tags = np.argmax(next_vars.npvalue(), 0)
        v_t = dy.max_dim(next_vars, 0)
        alphas = v_t + emission
        backpointers.append(best_tags)

    terminal_expr = alphas + dy.pick(transition, end_idx)
    best_tag = np.argmax(terminal_expr.npvalue())
    path_score = dy.pick(terminal_expr, best_tag)

    best_path = [best_tag]
    for bp_t in reversed(backpointers):
        best_tag = bp_t[best_tag]
        best_path.append(best_tag)
    _ = best_path.pop()
    best_path.reverse()
    return best_path, path_score

from itertools import chain
import numpy as np
import dynet as dy
from baseline.utils import crf_mask, lookup_sentence


class DynetModel(object):
    def __init__(self):
        super(DynetModel, self).__init__()
        self._pc = dy.ParameterCollection()

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


def optimizer(model, optim='sgd', eta=0.01, clip=None, mom=0.9, **kwargs):
    if 'lr' in kwargs:
        eta = kwargs['lr']
    print('Using eta [{:.4f}]'.format(eta))
    print('Using optim [{}]'.format(optim))
    if optim == 'adadelta':
        opt = dy.AdadeltaTrainer(model.pc)
    elif optim == 'adam':
        opt = dy.AdamTrainer(model.pc)
    elif optim == 'rmsprop':
        opt = dy.RMSPropTrainer(model.pc, learning_rate=eta)
    else:
        if mom == 0 or mom is None:
            opt = dy.SimpleSGDTrainer(model.pc, learning_rate=eta)
        else:
            print('Using mom {:.3f}'.format(mom))
            opt = dy.MomentumSGDTrainer(model.pc, learning_rate=eta, mom=mom)
    if clip is not None:
        opt.set_clip_threshold(clip)
    opt.set_sparse_updates(False)
    return opt


def Linear(osz, isz, pc, name="linear"):
    """
    :param osz: int
    :param isz: int
    :param pc: dy.ParameterCollection
    :param name: str
    """
    linear_pc = pc.add_subcollection(name=name)
    weight = linear_pc.add_parameters((osz, isz), name="weight")
    bias = linear_pc.add_parameters(osz, name="bias")

    def linear(input_):
        """
        :param input_: dy.Expression ((isz,), B)

        Returns:
            dy.Expression ((osz,), B)
        """
        output = dy.affine_transform([bias, weight, input_])
        return output

    return linear


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

    def highway(input_):
        for func, weight, bias in zip(funcs, weights, biases):
            proj = dy.rectify(func(input_))
            transform = dy.logistic(dy.affine_transform([bias, weight, input_]))
            input_ = dy.cmult(transform, proj) + dy.cmult(input_, 1 - transform)
        return input_

    return highway


def SkipConnection(funcs, *args, **kwargs):
    def skip(input_):
        for func in funcs:
            proj = func(input_)
            input_ = input_ + proj
        return input_
    return skip


def LSTM(osz, isz, pc, layers=1):
    """
    :param osz: int
    :param isz: int
    :param pc: dy.ParameterCollection
    :param layers: int
    """
    lstm = dy.VanillaLSTMBuilder(layers, isz, osz, pc)

    def encode(input_):
        """
        :param input_: List[dy.Expression] ((isz,), B)

        Returns:
            dy.Expression ((osz,), B)
        """
        state = lstm.initial_state()
        return state.transduce(input_)

    return encode


def TruncatedLSTM(
        osz, isz, pc, layers=1,
        dropout=None, batched=True
):
    """An LSTM that return the final state.
    :param osz: int
    :param isz: int
    :param pc: dy.ParameterCollection
    :param layers: int
    """
    lstm = dy.VanillaLSTMBuilder(layers, isz, osz, pc)

    def encode(input_, state=None, train=True):
        if state is not None:
            state = [dy.inputTensor(s, batched) for s in state]
        if train and dropout is not None:
            lstm.set_dropout(dropout)
        else:
            lstm.disable_dropout()
        lstm_state = lstm.initial_state(state)
        outputs = lstm_state.add_inputs(input_)
        last_state = outputs[-1].s()
        outputs = [out.h()[-1] for out in outputs]
        return outputs, last_state

    return encode


def BiLSTM(osz, isz, pc, layers=1):
    """
    :param osz: int
    :param isz: int
    :param pc: dy.ParameterCollection
    :param layers: int
    """
    lstm_forward = dy.VanillaLSTMBuilder(layers, isz, osz//2, pc)
    lstm_backward = dy.VanillaLSTMBuilder(layers, isz, osz//2, pc)

    def encode(input_):
        """
        :param input_: List[dy.Expression] ((isz,), B)

        Returns:
            dy.Expression ((osz,), B)
        """
        state_forward = lstm_forward.initial_state()
        state_backward = lstm_backward.initial_state()
        return state_forward.transduce(input_), state_backward.transduce(reversed(input_))

    return encode


def LSTMEncoder(osz, isz, pc, layers=1):
    """
    :param osz: int
    :param isz: int
    :param pc: dy.ParameterCollection
    :param layers: int
    """
    lstm = LSTM(osz, isz, pc, layers=layers)

    def encode(input_, lengths):
        states = dy.concatenate_cols(lstm(input_))
        final_states = dy.pick_batch(states, lengths, dim=1)
        return final_states

    return encode


def BiLSTMEncoder(osz, isz, pc, layers=1):
    """
    :param osz: int
    :param isz: int
    :param pc: dy.ParameterCollection
    :param layers: int
    """
    lstm = BiLSTM(osz, isz, pc, layers=layers)

    def encode(input_, lengths):
        forward, backward = lstm(input_)
        states = dy.concatenate_cols(forward)
        final_states_forward = dy.pick_batch(states, lengths, dim=1)
        states = dy.concatenate_cols(backward)
        final_states_backward = dy.pick_batch(states, lengths, dim=1)
        return dy.concatenate([final_states_forward, final_states_backward])

    return encode


def Convolution1d(fsz, cmotsz, dsz, pc, strides=(1, 1, 1, 1), name="conv"):
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
    glorot_bounds = 0.5 * np.sqrt(6 / (fan_in + fan_out))
    weight = conv_pc.add_parameters(
        (1, fsz, dsz, cmotsz),
        init=dy.UniformInitializer(glorot_bounds),
        name='weight'
    )
    bias = conv_pc.add_parameters((cmotsz), name="bias")

    def conv(input_):
        """Perform the 1D conv.

        :param input: dy.Expression ((1, T, dsz), B)

        Returns:
            dy.Expression ((cmotsz,), B)
        """
        c = dy.conv2d_bias(input_, weight, bias, strides, is_valid=False)
        activation = dy.rectify(c)
        mot = dy.reshape(dy.max_dim(activation, 1), (cmotsz,))
        return mot

    return conv


def ParallelConv(filtsz, cmotsz, dsz, pc, strides=(1, 1, 1, 1), name="parallel-conv"):
    if isinstance(cmotsz, int):
        cmotsz = [cmotsz] * len(filtsz)
    conv_pc = pc.add_subcollection(name=name)
    convs = [Convolution1d(fsz, cmot, dsz, conv_pc, strides, name="conv-{}".format(fsz)) for fsz, cmot in zip(filtsz, cmotsz)]

    def conv(input_):
        dims = tuple([1] + list(input_.dim()[0]))
        input_ = dy.reshape(input_, dims)
        mots = []
        for conv in convs:
            mots.append(conv(input_))
        return dy.concatenate(mots)

    return conv


def Embedding(vsz, dsz, pc,
              embedding_weight=None,
              finetune=False, dense=False, batched=False,
              name="embeddings"):
    """Create Embedding layer.

    :param vsz: int, The Vocab Size
    :param dsz: int, The Embeddings Size
    :param pc: dy.ParameterCollection
    :param embedding_weight: np.ndarray, Pretrained weights
    :param finetune: bool Should the vectors be updated
    :param dense: bool Should the result be a single matrix or a list of vectors
    :param batched: bool Is the input a batched operation
    """
    if embedding_weight is not None:
        if dense:
            vsz, dsz = embedding_weight.shape
            embedding_weight = np.reshape(embedding_weight, (vsz, 1, dsz))
        embeddings = pc.lookup_parameters_from_numpy(embedding_weight, name=name)
    else:
        shape = (vsz, dsz)
        if dense:
            shape = (vsz, 1, dsz)
        embeddings = pc.add_lookup_parameters(shape, name=name)

    def embed(input_):
        """Embed a sequence.

        :param input_: List[List[int]] (batched) or List[int] (normal)
            When batched the input should be a list over timesteps of lists of
            words (over a batch) (T, B). Otherwise it is a list of words over time (T)

        Returns:
            dy.Expression ((T, H), B) if dense (useful for conv encoders)
            List[dy.Expression] otherwise (used for RNNs)
        """
        lookup = dy.lookup_batch if batched else dy.lookup
        embedded = [lookup(embeddings, x, finetune) for x in input_]
        if dense:
            return dy.concatenate(embedded, d=0)
        return embedded

    return embed


def Attention(lstmsz, pc, name="attention"):
    """Vectorized Bahdanau Attention.

    :param lstmsz: int
    :param pc: dy.ParameterCollection
    """
    attn_pc = pc.add_subcollection(name=name)
    attention_w1 = attn_pc.add_parameters((lstmsz, lstmsz), name='encoder-projection')
    attention_w2 = attn_pc.add_parameters((lstmsz, lstmsz), name='decoder-projection')
    attention_v = attn_pc.add_parameters((1, lstmsz), name='attention-vector')

    def attention(encoder_vectors):
        """Compute the projected encoder vectors once per decoder.

        :param encoder_vectors: dy.Expression ((H, T), B)
            often from dy.concatenate_cols([lstm.transduce(embedded)])
        """
        projected_vectors = attention_w1 * encoder_vectors

        def attend(state):
            """Calculate the attention weighted sum of the encoder vectors.

            :param state: dy.Expression ((H,), B)
            """
            projected_state = attention_w2 * state
            non_lin = dy.tanh(dy.colwise_add(projected_vectors, projected_state))
            attention_scores = attention_v * non_lin
            attention_weights = dy.softmax(dy.transpose(attention_scores))
            # Encoder Vectors has data along the columns so a matmul with the
            # weights (which are also a column) creates a sum of encoder weighted
            # by attention weights
            output_vector = encoder_vectors * attention_weights
            return dy.reshape(output_vector, (lstmsz,))

        return attend

    return attention


class CRF(DynetModel):
    """Linear Chain CRF in Dynet."""

    def __init__(self, n_tags, pc=None, idxs=None, vocab=None, span_type=None, pad_idx=None):
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
        super(CRF, self).__init__()
        if pc is not None:
            self.pc = pc.add_subcollection(name="crf")
        if idxs is None:
            self.start_idx = n_tags
            self.end_idx = n_tags + 1
            self.n_tags = n_tags + 2
            self.add_ends = True
        else:
            self.start_idx, self.end_idx = idxs
            self.n_tags = n_tags
            self.add_ends = False
        self.mask = None
        if vocab is not None:
            assert span_type is not None, "To mask transitions you need to provide a tagging span_type, choices are `IOB`, `BIO` (or `IOB2`), and `IOBES`"
            if idxs is None:
                vocab = vocab.copy()
                vocab['<GO>'] = self.start_idx
                vocab['<EOS>'] = self.end_idx
            self.mask = crf_mask(vocab, span_type, self.start_idx, self.end_idx, pad_idx)
            self.inv_mask = (self.mask == 0) * -1e4

        self.transitions_p = self.pc.add_parameters((self.n_tags, self.n_tags), name="transition")

    @property
    def transitions(self):
        if self.mask is not None:
            return dy.cmult(self.transitions_p, dy.inputTensor(self.mask)) + dy.inputTensor(self.inv_mask)
        return self.transitions_p

    @staticmethod
    def _prep_input(emissions):
        """Append scores for start and end to the emission.

        :param emissions: List[dy.Expression ((H,), B)]

        Returns:
            List[dy.Expression ((H + 2,), B)]
        """
        return [dy.concatenate([e, dy.inputVector([-1e4, -1e4])], d=0) for e in emissions]

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
        if self.add_ends:
            emissions = CRF._prep_input(emissions)
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
        if self.add_ends:
            emissions = CRF._prep_input(emissions)
        backpointers = []
        transitions = self.transitions

        inits = [-1e4] * self.n_tags
        inits[self.start_idx] = 0
        alphas = dy.inputVector(inits)

        for emission in emissions:
            next_vars = dy.colwise_add(dy.transpose(transitions), alphas)
            best_tags = np.argmax(next_vars.npvalue(), 0)
            v_t = dy.max_dim(next_vars, 0)
            alphas = v_t + emission
            backpointers.append(best_tags)

        terminal_expr = alphas + dy.pick(transitions, self.end_idx)
        best_tag = np.argmax(terminal_expr.npvalue())
        path_score = dy.pick(terminal_expr, best_tag)

        best_path = [best_tag]
        for bp_t in reversed(backpointers):
            best_tag = bp_t[best_tag]
            best_path.append(best_tag)
        _ = best_path.pop()
        best_path.reverse()
        return best_path, path_score


def show_examples_dynet(model, es, rlut1, rlut2, embed2, mxlen, sample, prob_clip, max_examples, reverse):
    si = np.random.randint(0, len(es))

    batch_dict = es[si]

    src_array = batch_dict['src']
    tgt_array = batch_dict['dst']
    src_len = batch_dict['src_len']
    if max_examples > 0:
        max_examples = min(max_examples, src_array.shape[0])
        src_array = src_array[0:max_examples]
        tgt_array = tgt_array[0:max_examples]
        src_len = src_len[0:max_examples]

    for src_len_i, src_i, tgt_i in zip(src_len, src_array, tgt_array):

        print('========================================================================')
        src_len_i = np.array(src_len_i, ndmin=1)
        sent = lookup_sentence(rlut1, src_i, reverse=reverse)
        print('[OP] %s' % sent)
        sent = lookup_sentence(rlut2, tgt_i)
        print('[Actual] %s' % sent)
        src_dict = {'src': src_i.reshape(1, -1), 'src_len': src_len_i.reshape(1, -1)}
        dst_i = model.run(src_dict)[0]
        sent = lookup_sentence(rlut2, dst_i)
        print('Guess: %s' % sent)
        print('------------------------------------------------------------------------')


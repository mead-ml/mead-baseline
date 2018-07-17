from baseline.model import create_seq2seq_model, load_seq2seq_model, EncoderDecoder
from baseline.dy.dynety import *
from baseline.utils import topk


class Seq2SeqModel(DynetModel, EncoderDecoder):

    def __init__(self, embeddings_in, embeddings_out, **kwargs):
        super(Seq2SeqModel, self).__init__()
        self.train = True
        self.hsz = kwargs['hsz']
        nlayers = kwargs['layers']
        self.rnntype = kwargs['rnntype']
        print(self.rnntype)
        self.pdrop = kwargs.get('dropout', 0.5)
        self.nc = embeddings_out.vsz + 1
        self.embed_in = Embedding(embeddings_in.vsz + 1, embeddings_in.dsz, self.pc, embedding_weight=embeddings_in.weights, finetune=True, batched=True)
        self.embed_out = Embedding(self.nc, embeddings_out.dsz, self.pc, embedding_weight=embeddings_out.weights, finetune=True, batched=True)

        if self.rnntype == 'blstm':
            self.lstm_forward = dy.VanillaLSTMBuilder(nlayers, embeddings_in.dsz, self.hsz//2, self.pc)
            self.lstm_backward = dy.VanillaLSTMBuilder(nlayers, embeddings_in.dsz, self.hsz//2, self.pc)
        else:
            self.lstm_forward = dy.VanillaLSTMBuilder(nlayers, embeddings_in.dsz, self.hsz, self.pc)
            self.lstm_backward = None

        # TODO: hsz + dsz x for input feeding!
        self.decoder_rnn = dy.VanillaLSTMBuilder(nlayers, self.hsz, embeddings_out.dsz, self.pc)
        self.preds = Linear(self.nc, embeddings_out.dsz, self.pc)
        self.vocab1 = embeddings_in.vocab
        self.vocab2 = embeddings_out.vocab
        self.beam_sz = 1

    def encode_rnn(self, embed_in_seq):

        state_forward = self.lstm_forward.initial_state()
        outputs = state_forward.add_inputs(embed_in_seq)
        forward_state = outputs[-1].s()
        forward = [out.h()[-1] for out in outputs]
        if self.lstm_backward is not None:
            state_backward = self.lstm_backward.initial_state()
            outputs = state_backward.add_inputs(reversed(embed_in_seq))
            backward_state = outputs[-1].s()
            backward = [out.h()[-1] for out in outputs]
            output = [dy.concatenate([f, b]) for f, b in zip(forward, backward)]
            c = dy.concatenate([forward_state[0], backward_state[0]])
            h = dy.concatenate([forward_state[1], backward_state[1]])
            hidden = (c, h)
        else:
            output = forward
            hidden = forward_state

        return output, hidden

    def save(self, file_name):
        self.pc.save(file_name)
        return self

    def load(self, file_name):
        self.pc.populate(file_name)
        return self

    def get_src_vocab(self):
        return self.vocab1

    def get_dst_vocab(self):
        return self.vocab2

    def dropout(self, input_):
        if self.train:
            return dy.dropout(input_, self.pdrop)
        return input_

    @classmethod
    def create(cls, input_embeddings, output_embeddings, **kwargs):

        model = cls(input_embeddings, output_embeddings, **kwargs)
        print(model)
        return model

    def make_input(self, batch_dict, **kwargs):
        src = batch_dict['src'].T
        src_len = batch_dict['src_len'].T
        dst = batch_dict['dst'].T
        tgt = dst[1:]
        return src, dst[:-1], src_len, tgt

    # Input better be xch, x
    def forward(self, input):
        src = input[0]
        dst = input[1]
        src_len = input[2]
        rnn_enc_seq, final_encoder_state = self.encode(src, src_len)
        return self.decode(rnn_enc_seq, src_len, final_encoder_state, dst)

    def encode(self, src, src_len):
        embed_in_seq = self.embed_in(src)
        output, hidden = self.encode_rnn(embed_in_seq)
        return output, hidden

    def input_i(self, embed_i, output_i):
        #return embed_i
        return embed_i

    def bridge(self, final_encoder_state, context):
        return final_encoder_state, context[-1]

    def _attn(self, context_mx):
        def ident(x):
            return x
        return ident

    def decode_rnn(self, embed_out_seq, rnn_state, output_i, attn_fn):
        output = []
        for i, embed_i in enumerate(embed_out_seq):
            embed_i = self.input_i(embed_i, output_i)
            rnn_state = rnn_state.add_input(embed_i)
            rnn_output_i = rnn_state.output()
            rnn_output_i = self.dropout(rnn_output_i)
            output_i = attn_fn(rnn_output_i)
            output += [output_i]
        return output

    def decode(self, context, src_len, final_encoder_state, dst):
        h_i, output_i = self.bridge(final_encoder_state, context)
        context_mx = dy.concatenate_cols(context)
        rnn_state = self.decoder_rnn.initial_state(h_i)
        attn_fn = self._attn(context_mx)
        embed_out_seq = self.embed_out(dst)
        output = []
        for i, embed_i in enumerate(embed_out_seq):
            embed_i = self.input_i(embed_i, output_i)
            rnn_state = rnn_state.add_input(embed_i)
            rnn_output_i = rnn_state.output()
            rnn_output_i = self.dropout(rnn_output_i)
            output_i = attn_fn(rnn_output_i)
            output += [self.preds(output_i)]

        return output

    def prediction(self, output):
        p = self.preds(output)
        lsm = dy.log_softmax(p)
        return lsm

    # B x K x T and here T is a list
    def run(self, batch_dict, **kwargs):
        src = batch_dict['src']
        src_len = batch_dict['src_len']
        if type(src_len) == int or type(src_len) == np.int64:
            src_len = np.array([src_len])
        batch = []
        for src_i, src_len_i in zip(src, src_len):
            #batch += [self.greedy_decode(src_i.reshape(-1, 1), src_len_i.reshape(-1, 1))]
            best = self.beam_decode(src_i.reshape(-1, 1), src_len_i.reshape(-1, 1), K=kwargs.get('beam', 1))[0][0]
            batch += [best]

        return batch

    def greedy_decode(self, src, src_len):
        GO = self.vocab2['<GO>']
        EOS = self.vocab2['<EOS>']
        dy.renew_cg()
        rnn_enc_seq, final_encoder_state = self.encode(src, src_len)
        context_mx = dy.concatenate_cols(rnn_enc_seq)
        # rnn_state = self.decoder_rnn.initial_state(final_encoder_state)
        rnn_state = self.decoder_rnn.initial_state(final_encoder_state)
        attn_fn = self._attn(context_mx)

        output_i = rnn_enc_seq[-1]  # zeros?!
        output = [GO]
        for i in range(100):
            dst_last = np.array(output[-1]).reshape(1, 1)
            embed_i = self.embed_out([dst_last])[-1]
            embed_i = self.input_i(embed_i, output_i)
            rnn_state = rnn_state.add_input(embed_i)
            rnn_output_i = rnn_state.output()
            output_i = attn_fn(rnn_output_i)
            output_i = np.argmax(self.preds(output_i).npvalue())
            output += [output_i]
            if output_i == EOS:
                break

        return output

    def beam_decode(self, src, src_len, K=2):

        GO = self.vocab2['<GO>']
        EOS = self.vocab2['<EOS>']

        paths = [[GO] for _ in range(K)]
        # Which beams are done?
        done = np.array([False] * K)
        scores = np.array([0.0]*K)
        dy.renew_cg()
        rnn_enc_seq, (c, h) = self.encode(src, src_len)
        context_mx = dy.concatenate_cols(rnn_enc_seq)

        # To vectorize, we need to expand along the batch dimension, K times
        final_encoder_state_k = dy.concatenate_to_batch([c]*K), dy.concatenate_to_batch([h]*K)
        rnn_state = self.decoder_rnn.initial_state(final_encoder_state_k)
        attn_fn = self._attn(context_mx)

        output_i = dy.concatenate_to_batch([rnn_enc_seq[-1]]*K)
        for i in range(100):
            dst_last = np.array([path[-1] for path in paths]).reshape(1, K)
            embed_i = self.embed_out(dst_last)[-1]
            embed_i = self.input_i(embed_i, output_i)
            rnn_state = rnn_state.add_input(embed_i)
            rnn_output_i = rnn_state.output()
            output_i = attn_fn(rnn_output_i)
            wll = self.prediction(output_i).npvalue()  # (V,) K
            V = wll.shape[0]
            if i > 0:
                expanded_history = scores.reshape(scores.shape + (1,))  # scores = K
                # TODO: dont add anything when the beam is done
                sll = wll.T + expanded_history
            else:
                sll = wll.T

            flat_sll = sll.reshape(-1)

            bests = topk(K, flat_sll)
            best_idx_flat = np.array(list(bests.keys()))
            best_beams = best_idx_flat // V
            best_idx = best_idx_flat % V

            new_paths = []
            new_done = []
            new_c = []
            new_h = []
            c, h = rnn_state.s()

            for j, best_flat in enumerate(best_idx_flat):
                beam_id = best_beams[j]
                best_word = best_idx[j]
                if best_word == EOS:
                    done[j] = True
                new_done.append(done[beam_id])
                new_paths.append(paths[beam_id] + [best_word])
                scores[j] = bests[best_flat]
                new_c.append(dy.pick_batch_elem(c, beam_id))
                new_h.append(dy.pick_batch_elem(h, beam_id))

            done = np.array(new_done)
            new_c = dy.concatenate_to_batch(new_c)
            new_h = dy.concatenate_to_batch(new_h)
            paths = new_paths
            # Now comes the hard part, fix the hidden units
            # Copy the beam states of the winners
            rnn_state = self.decoder_rnn.initial_state((new_c, new_h))

        return [p[1:] for p in paths], scores


class Seq2SeqAttnModel(Seq2SeqModel):

    def __init__(self, embeddings_in, embeddings_out, **kwargs):
        super(Seq2SeqAttnModel, self).__init__(embeddings_in, embeddings_out, **kwargs)
        self.attn_module = Attention(kwargs.get('hsz'), self.pc)

    def _attn(self, context):
        return self.attn_module(context)

    def input_i(self, embed_i, output_i):
        ##return dy.concatenate([embed_i, output_i])
        return embed_i

BASELINE_SEQ2SEQ_MODELS = {
    'default': Seq2SeqModel.create,
    'attn': Seq2SeqAttnModel.create
}


def create_model(embeddings_in, embeddings_out, **kwargs):
    lm = create_seq2seq_model(BASELINE_SEQ2SEQ_MODELS, embeddings_in, embeddings_out, **kwargs)
    return lm


def load_model(modelname, **kwargs):
    return load_seq2seq_model(BASELINE_SEQ2SEQ_MODELS, modelname, **kwargs)

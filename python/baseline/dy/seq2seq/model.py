import numpy as np
from baseline.model import EncoderDecoderModel, register_model
from baseline.dy.dynety import *
from baseline.utils import topk, Offsets


@register_model(task='seq2seq', name='default')
class Seq2SeqModel(DynetModel, EncoderDecoderModel):

    def __init__(self, embeddings_in, embeddings_out, **kwargs):
        super(Seq2SeqModel, self).__init__(kwargs['pc'])
        self.train = True
        self.hsz = kwargs['hsz']
        layers = kwargs['layers']
        self.rnntype = kwargs['rnntype']
        self.pdrop = kwargs.get('dropout', 0.5)
        self.nc = embeddings_out.vsz
        src_dsz = self.init_embed(embeddings_in)
        self.tgt_embedding = embeddings_out
        self.src_lengths_key = kwargs.get('src_lengths_key')

        if self.rnntype == 'blstm':
            self.lstm_forward = dy.VanillaLSTMBuilder(layers, src_dsz, self.hsz//2, self.pc)
            self.lstm_backward = dy.VanillaLSTMBuilder(layers, src_dsz, self.hsz//2, self.pc)
        else:
            self.lstm_forward = dy.VanillaLSTMBuilder(layers, src_dsz, self.hsz, self.pc)
            self.lstm_backward = None

        # TODO: hsz + dsz x for input feeding!
        out_dsz = self.tgt_embedding.dsz
        self.decoder_rnn = dy.VanillaLSTMBuilder(layers, self.hsz, out_dsz, self.pc)
        self.preds = Linear(self.nc, out_dsz, self.pc)
        self.beam_sz = 1

    def init_embed(self, embeddings):
        dsz = 0
        self.embeddings = embeddings
        for embedding in self.embeddings.values():
            dsz += embedding.get_dsz()
        return dsz

    def encoder(self, embed_in_seq, src_len):
        forward, forward_state = rnn_forward_with_state(self.lstm_forward, embed_in_seq, src_len)
        if self.lstm_backward is not None:
            backward, backward_state = rnn_forward_with_state(self.lstm_backward, embed_in_seq, src_len, backward=True)

            output = [dy.concatenate([f, b]) for f, b in zip(forward, backward)]
            hidden = [dy.concatenate([f, b]) for f, b in zip(forward_state, backward_state)]
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

    def dropout(self, input_):
        if self.train:
            return dy.dropout(input_, self.pdrop)
        return input_

    @classmethod
    def create(cls, input_embeddings, output_embeddings, **kwargs):
        model = cls(input_embeddings, output_embeddings, **kwargs)
        print(model)
        return model

    def embed(self, batch_dict):
        all_embeddings_lists = []
        for k, embedding in self.embeddings.items():
            all_embeddings_lists.append(embedding.encode(batch_dict[k]))

        embeddings = dy.concatenate(all_embeddings_lists, d=1)
        return embeddings

    def make_input(self, batch_dict):
        example_dict = dict({})
        for k in self.embeddings.keys():
            example_dict[k] = batch_dict[k].T

        lengths = batch_dict[self.src_lengths_key]
        example_dict['src_len'] = lengths.T

        if 'tgt' in batch_dict:
            tgt = batch_dict['tgt'].T
            example_dict['dst'] = tgt[:-1]
            example_dict['tgt'] = tgt[1:]
        return example_dict

    def forward(self, batch_dict):
        rnn_enc_seq, final_encoder_state = self.encode(batch_dict)
        dst = batch_dict['dst']
        return self.decode(rnn_enc_seq, final_encoder_state, dst)

    def encode(self, example_dict):
        embed_in_seq = self.embed(example_dict)
        src_len = example_dict['src_len']
        output, hidden = self.encoder(embed_in_seq, src_len)
        return output, hidden

    def input_i(self, embed_i, output_i):
        return embed_i

    def bridge(self, final_encoder_state, context):
        return final_encoder_state, context[-1]

    def attn(self, context_mx):
        def ident(x):
            return x
        return ident

    def decoder(self, embed_out_seq, rnn_state, output_i, attn_fn):
        output = []
        for i, embed_i in enumerate(embed_out_seq):
            embed_i = self.input_i(embed_i, output_i)
            rnn_state = rnn_state.add_input(embed_i)
            rnn_output_i = rnn_state.output()
            rnn_output_i = self.dropout(rnn_output_i)
            output_i = attn_fn(rnn_output_i)
            output.append(output_i)
        return output

    def decode(self, context, final_encoder_state, dst):
        h_i, output_i = self.bridge(final_encoder_state, context)
        context_mx = dy.concatenate_cols(context)
        rnn_state = self.decoder_rnn.initial_state(h_i)
        attn_fn = self.attn(context_mx)
        embed_out_seq = self.tgt_embedding.encode(dst)
        output = []
        for i, embed_i in enumerate(embed_out_seq):
            embed_i = self.input_i(embed_i, output_i)
            rnn_state = rnn_state.add_input(embed_i)
            rnn_output_i = rnn_state.output()
            rnn_output_i = self.dropout(rnn_output_i)
            output_i = attn_fn(rnn_output_i)
            output.append(self.preds(output_i))

        return output

    def prediction(self, output):
        p = self.preds(output)
        lsm = dy.log_softmax(p)
        return lsm

    # B x K x T and here T is a list
    def predict(self, batch_dict, beam=1, **kwargs):
        self.train = False
        # Bit of a hack
        src_field = self.src_lengths_key.split('_')[0]
        B = batch_dict[src_field].shape[0]
        batch = []
        for b in range(B):
            example = dict({})
            for k, value in batch_dict.items():
                example[k] = value[b].reshape((1,) + value[b].shape)
            inputs = self.make_input(example)
            batch.append(self.beam_decode(inputs, beam, kwargs.get('mxlen', 100))[0])

        return batch

    def greedy_decode(self, inputs, mxlen=100):
        dy.renew_cg()
        rnn_enc_seq, final_encoder_state = self.encode(inputs)
        context_mx = dy.concatenate_cols(rnn_enc_seq)
        # rnn_state = self.decoder_rnn.initial_state(final_encoder_state)
        rnn_state = self.decoder_rnn.initial_state(final_encoder_state)
        attn_fn = self.attn(context_mx)

        output_i = rnn_enc_seq[-1]  # zeros?!
        output = [Offsets.GO]
        for i in range(mxlen):
            dst_last = np.array(output[-1]).reshape(1, 1)
            embed_i = self.tgt_embedding([dst_last])[-1]
            embed_i = self.input_i(embed_i, output_i)
            rnn_state = rnn_state.add_input(embed_i)
            rnn_output_i = rnn_state.output()
            output_i = attn_fn(rnn_output_i)
            output_i = np.argmax(self.preds(output_i).npvalue())
            output.append(output_i)
            if output_i == Offsets.EOS:
                break

        return output

    def beam_decode(self, inputs, K, mxlen=100):
        dy.renew_cg()

        paths = [[GO] for _ in range(K)]
        # Which beams are done?
        done = np.array([False] * K)
        scores = np.array([0.0]*K)
        dy.renew_cg()
        rnn_enc_seq, hidden = self.encode(inputs)
        context_mx = dy.concatenate_cols(rnn_enc_seq)
        # To vectorize, we need to expand along the batch dimension, K times
        final_encoder_state_k = (dy.concatenate_to_batch([h]*K) for h in hidden)
        num_states = len(hidden)
        rnn_state = self.decoder_rnn.initial_state(final_encoder_state_k)
        attn_fn = self.attn(context_mx)

        output_i = dy.concatenate_to_batch([rnn_enc_seq[-1]]*K)
        for i in range(mxlen):
            dst_last = np.array([path[-1] for path in paths]).reshape(1, K)
            embed_i = self.tgt_embedding.encode(dst_last)[-1]
            embed_i = self.input_i(embed_i, output_i)
            rnn_state = rnn_state.add_input(embed_i)
            rnn_output_i = rnn_state.output()
            output_i = attn_fn(rnn_output_i)
            wll = self.prediction(output_i).npvalue()  # (V,) K
            V = wll.shape[0]
            if i > 0:
                expanded_history = scores.reshape(scores.shape + (1,))  # scores = K
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

            hidden = rnn_state.s()
            # For each hidden state
            new_hidden = [[] for _ in range(num_states)]
            for j, best_flat in enumerate(best_idx_flat):
                beam_id = best_beams[j]
                best_word = best_idx[j]
                if best_word == Offsets.EOS:
                    done[j] = True
                new_done.append(done[beam_id])
                new_paths.append(paths[beam_id] + [best_word])
                scores[j] = bests[best_flat]
                # For each path, we need to pick that out and add it to the hiddens
                # This will be (c1, c2, ..., h1, h2, ...)
                for h_i, h in enumerate(hidden):
                    new_hidden[h_i].append(dy.pick_batch_elem(h, beam_id))

            done = np.array(new_done)
            new_hidden = [dy.concatenate_to_batch(new_h) for new_h in new_hidden]
            paths = new_paths
            # Now comes the hard part, fix the hidden units
            # Copy the beam states of the winners
            rnn_state = self.decoder_rnn.initial_state(new_hidden)

        return [p[1:] for p in paths], scores


@register_model(task='seq2seq', name='attn')
class Seq2SeqAttnModel(Seq2SeqModel):

    def __init__(self, embeddings_in, embeddings_out, **kwargs):
        super(Seq2SeqAttnModel, self).__init__(embeddings_in, embeddings_out, **kwargs)
        self.attn_module = Attention(kwargs.get('hsz'), self.pc)

    def attn(self, context):
        return self.attn_module(context)

    def input_i(self, embed_i, output_i):
        ##return dy.concatenate([embed_i, output_i])
        return embed_i

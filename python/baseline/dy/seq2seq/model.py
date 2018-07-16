from baseline.model import create_seq2seq_model, load_seq2seq_model, EncoderDecoder
from baseline.dy.dynety import *


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

        # hsz + dsz x for input feeding!
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
            output = [dy.concatenate([c, h]) for c, h in zip(forward, backward)]
            c = dy.concatenate([forward_state[0], backward_state[0]])
            h = dy.concatenate([forward_state[1], backward_state[1]])
            hidden = (c, h)
        else:
            output = forward
            hidden = forward_state

        ##print('HH', [h.dim() for h in hidden])
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
        #print([h.dim() for h in h_i])
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
            output += [self.prediction(output_i)]

        return output

    def prediction(self, output):
        return self.preds(output)

    # B x K x T and here T is a list
    def run(self, batch_dict, **kwargs):
        src = batch_dict['src']
        src_len = batch_dict['src_len']
        if type(src_len) == int or type(src_len) == np.int64:
            src_len = np.array([src_len])
        batch = []
        for src_i, src_len_i in zip(src, src_len):
            batch += [self.greedy_decode(src_i.reshape(-1, 1), src_len_i.reshape(-1, 1))]

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
            output_i = np.argmax(self.prediction(output_i).npvalue())
            output += [output_i]
            if output_i == EOS:
                break

        return output


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

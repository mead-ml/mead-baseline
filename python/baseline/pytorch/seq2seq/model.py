import torch
import torch.nn as nn
from torch.autograd import Variable
from baseline.pytorch.torchy import *
from baseline.model import EncoderDecoder, load_seq2seq_model, create_seq2seq_model


class Seq2SeqBase(nn.Module, EncoderDecoder):

    def __init__(self, embeddings_in, embeddings_out):
        super(Seq2SeqBase, self).__init__()
        self.embed_in = pytorch_embedding(embeddings_in)
        self.embed_out = pytorch_embedding(embeddings_out)
        self.nc = embeddings_out.vsz + 1
        self.vocab1 = embeddings_in.vocab
        self.vocab2 = embeddings_out.vocab
        self.beam_sz = 1

    def get_src_vocab(self):
        return self.vocab1

    def get_dst_vocab(self):
        return self.vocab2

    def save(self, model_file):
        torch.save(self, model_file)

    def create_loss(self):
        return SequenceCriterion()

    @classmethod
    def load(cls, outname, **kwargs):
        model = torch.load(outname)
        return model

    @classmethod
    def create(cls, input_embeddings, output_embeddings, **kwargs):

        model = cls(input_embeddings, output_embeddings, **kwargs)
        print(model)
        return model

    def make_input(self, batch_dict):
        src = batch_dict['src']
        tgt = batch_dict['dst']

        dst = tgt[:, :-1]
        tgt = tgt[:, 1:]
        if self.gpu:
            src = src.cuda()
            dst = dst.cuda()
            tgt = tgt.cuda()
        return Variable(src), Variable(dst), Variable(tgt)

    # Input better be xch, x
    def forward(self, input):
        rnn_enc_seq, final_encoder_state = self.encode(input[0])
        return self.decode(rnn_enc_seq, final_encoder_state, input[1])

    def encode(self, src):
        src = src.transpose(0, 1).contiguous()
        embed_in_seq = self.embed_in(src)
        return self.encoder_rnn(embed_in_seq)

    def input_i(self, embed_i, output_i):
        pass

    def bridge(self, final_encoder_state, context):
        pass

    def attn(self, output_t, context):
        pass

    def decode_rnn(self, context, h_i, output_i, dst):
        embed_out_seq = self.embed_out(dst)
        context_transpose = context.transpose(0, 1)
        outputs = []

        for i, embed_i in enumerate(embed_out_seq.split(1)):
            embed_i = self.input_i(embed_i, output_i)
            output_i, h_i = self.decoder_rnn(embed_i, h_i)
            output_i = self.attn(output_i, context_transpose)
            output_i = self.dropout(output_i)
            outputs += [output_i]

        outputs = torch.stack(outputs)
        return outputs, h_i

    def decode(self, context, final_encoder_state, dst):
        dst = dst.transpose(0, 1).contiguous()

        h_i, output_i = self.bridge(final_encoder_state, context)
        output, _ = self.decode_rnn(context, h_i, output_i, dst)
        pred = self.prediction(output)
        # Reform batch as (T x B, D)
        #pred = self.probs(self.preds(output.view(output.size(0)*output.size(1),
        #                                         -1)))
        # back to T x B x H -> B x T x H
        #pred = pred.view(output.size(0), output.size(1), -1)
        return pred.transpose(0, 1).contiguous()

    def prediction(self, output):
        # Reform batch as (T x B, D)
        pred = self.probs(self.preds(output.view(output.size(0)*output.size(1),
                                                 -1)))
        # back to T x B x H -> B x T x H
        pred = pred.view(output.size(0), output.size(1), -1)
        return pred


class Seq2SeqModel(Seq2SeqBase):

    def __init__(self, embeddings_in, embeddings_out, **kwargs):
        super(Seq2SeqModel, self).__init__(embeddings_in, embeddings_out)

        self.hsz = kwargs['hsz']
        nlayers = kwargs['layers']
        rnntype = kwargs['rnntype']
        pdrop = kwargs.get('dropout', 0.5)
        enc_hsz = self.hsz
        if rnntype == 'blstm' or rnntype == 'bgru':
            enc_hsz = enc_hsz // 2
        dsz = embeddings_in.dsz
        self.gpu = kwargs.get('gpu', True)
        self.dropout = nn.Dropout(pdrop)
        self.encoder_rnn = pytorch_rnn(dsz, enc_hsz, rnntype, nlayers, pdrop)
        self.preds = nn.Linear(self.hsz, self.nc)
        self.decoder_rnn = pytorch_rnn_cell(dsz, self.hsz, rnntype, nlayers, pdrop)
        self.probs = nn.LogSoftmax()

    def input_i(self, embed_i, output_i):
        return embed_i.squeeze(0)

    def bridge(self, final_encoder_state, context):
        return final_encoder_state, None

    def attn(self, output_t, context):
        return output_t


class Seq2SeqAttnModel(Seq2SeqBase):

    def __init__(self, embeddings_in, embeddings_out, **kwargs):
        super(Seq2SeqAttnModel, self).__init__(embeddings_in, embeddings_out)
        self.hsz = kwargs['hsz']
        nlayers = kwargs['layers']
        rnntype = kwargs['rnntype']
        pdrop = kwargs.get('dropout', 0.5)
        enc_hsz = self.hsz
        if rnntype == 'lstm' or rnntype == 'gru':
            enc_hsz = enc_hsz // 2
        dsz = embeddings_in.dsz
        self.gpu = kwargs.get('gpu', True)
        self.encoder_rnn = pytorch_rnn(dsz, enc_hsz, rnntype, nlayers, pdrop)
        self.dropout = nn.Dropout(pdrop)
        self.decoder_rnn = pytorch_rnn_cell(self.hsz + dsz, self.hsz, rnntype, nlayers, pdrop)
        self.preds = nn.Linear(self.hsz, self.nc)
        self.probs = nn.LogSoftmax()
        self.output_to_attn = nn.Linear(self.hsz, self.hsz, bias=False)
        self.attn_softmax = nn.Softmax()
        self.attn_out = nn.Linear(2 * self.hsz, self.hsz, bias=False)
        self.attn_tanh = pytorch_activation("tanh")
        self.nlayers = nlayers

    def attn(self, output_t, context):
        # Output(t) = B x H x 1
        # Context = B x T x H
        # a = B x T x 1
        a = torch.bmm(context, self.output_to_attn(output_t).unsqueeze(2))
        a = self.attn_softmax(a.squeeze(2))
        # a = B x T
        # Want to apply over context, scaled by a
        # (B x 1 x T) (B x T x H) = (B x 1 x H)
        a = a.view(a.size(0), 1, a.size(1))
        combined = torch.bmm(a, context).squeeze(1)
        combined = torch.cat([combined, output_t], 1)
        combined = self.attn_tanh(self.attn_out(combined))
        return combined

    def bridge(self, final_encoder_state, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.hsz)
        context_zeros = Variable(context.data.new(*h_size).zero_(), requires_grad=False)
        if type(final_encoder_state) is tuple:
            s1, s2 = final_encoder_state
            return (s1*0, s2*0), context_zeros
        else:
            return final_encoder_state * 0, context_zeros

    def input_i(self, embed_i, output_i):
        embed_i = embed_i.squeeze(0)
        return torch.cat([embed_i, output_i], 1)


BASELINE_SEQ2SEQ_MODELS = {
    'default': Seq2SeqModel.create,
    'attn': Seq2SeqAttnModel.create
}
BASELINE_SEQ2SEQ_LOADERS = {
    'default': Seq2SeqModel.load,
    'attn': Seq2SeqAttnModel.create
}


def create_model(src_vocab_embed, dst_vocab_embed, **kwargs):
    model = create_seq2seq_model(BASELINE_SEQ2SEQ_MODELS, src_vocab_embed, dst_vocab_embed, **kwargs)
    return model


def load_model(modelname, **kwargs):
    return load_seq2seq_model(BASELINE_SEQ2SEQ_LOADERS, modelname, **kwargs)

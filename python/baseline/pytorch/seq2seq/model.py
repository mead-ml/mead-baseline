import torch
import torch.nn as nn
from torch.autograd import Variable
from baseline.pytorch.torchy import *
from baseline.model import EncoderDecoder


class Seq2SeqBase(nn.Module, EncoderDecoder):

    def __init__(self):
        super(Seq2SeqBase, self).__init__()

    def save(self, model_file):
        torch.save(self, model_file)

    def create_loss(self):
        return SequenceCriterion(self.nc)

    @staticmethod
    def load(model_file):
        return torch.load(model_file)

    # Input better be xch, x
    def forward(self, input):
        rnn_enc_seq, final_encoder_state = self.encode(input[0])
        return self.decode(rnn_enc_seq, final_encoder_state, input[1])

    def encode(self, src):
        if self.batchfirst is True:
            src = src.transpose(0, 1).contiguous()

        embed_in_seq = self.embed_in(src)
        return self.encoder_rnn(embed_in_seq)

    def input_i(self, embed_i, output_i):
        pass

    def bridge(self, final_encoder_state, context):
        pass

    def attn(self, output_t, context):
        pass

    def decode(self, context, final_encoder_state, dst):
        if self.batchfirst is True:
            dst = dst.transpose(0, 1).contiguous()

        embed_out_seq = self.embed_out(dst)
        h_i, output_i = self.bridge(final_encoder_state, context)
        context_transpose = context.t()
        outputs = []

        for i, embed_i in enumerate(embed_out_seq.split(1)):
            embed_i = self.input_i(embed_i, output_i)
            output_i, h_i = self.decoder_rnn(embed_i, h_i)
            output_i = self.attn(output_i, context_transpose)
            output_i = self.dropout(output_i)
            outputs += [output_i]

        output = torch.stack(outputs)

        # Reform batch as (T x B, D)
        pred = self.probs(self.preds(output.view(output.size(0)*output.size(1),
                                                 -1)))
        # back to T x B x H -> B x T x H
        pred = pred.view(output.size(0), output.size(1), -1)
        return pred.transpose(0, 1).contiguous() if self.batchfirst else pred


class Seq2SeqModel(Seq2SeqBase):

    # TODO: Add more dropout, BN
    def __init__(self, embed1, embed2, hsz, nlayers, rnntype, batchfirst=True, pdrop=0.5):
        super(Seq2SeqModel, self).__init__()
        dsz = embed1.dsz

        self.dropout = nn.Dropout(pdrop)
        self.embed_in = pytorch_embedding(embed1)
        self.embed_out = pytorch_embedding(embed2)
        self.nc = embed2.vsz + 1            
        self.encoder_rnn = pytorch_rnn(dsz, hsz, rnntype, nlayers, pdrop)
        self.preds = nn.Linear(hsz, self.nc)
        self.decoder_rnn = pytorch_rnn_cell(dsz, hsz, rnntype, nlayers, pdrop)
        self.batchfirst = batchfirst
        self.probs = nn.LogSoftmax()


    def input_i(self, embed_i, output_i):
        return embed_i.squeeze(0)

    def bridge(self, final_encoder_state, context):
        return final_encoder_state, None

    def attn(self, output_t, context):
        return output_t


class Seq2SeqAttnModel(Seq2SeqBase):

    def __init__(self, embed1, embed2, hsz, nlayers, rnntype, batchfirst=True, pdrop=0.5):
        super(Seq2SeqAttnModel, self).__init__()
        dsz = embed1.dsz
        self.embed_in = pytorch_embedding(embed1)
        self.embed_out = pytorch_embedding(embed2)
        self.nc = embed2.vsz + 1            
        self.encoder_rnn = pytorch_rnn(dsz, hsz, rnntype, nlayers, pdrop)
        self.dropout = nn.Dropout(pdrop)
        self.decoder_rnn = pytorch_rnn_cell(hsz + dsz, hsz, rnntype, nlayers, pdrop)
        self.preds = nn.Linear(hsz, self.nc)
        self.batchfirst = batchfirst
        self.probs = nn.LogSoftmax()
        self.output_to_attn = nn.Linear(hsz, hsz, bias=False)
        self.attn_softmax = nn.Softmax()
        self.attn_out = nn.Linear(2*hsz, hsz, bias=False)
        self.attn_tanh = nn.Tanh()
        self.nlayers = nlayers
        self.hsz = hsz

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


def create_model(embedding1, embedding2, **kwargs):
    hsz = int(kwargs['hsz'])
    attn = bool(kwargs.get('attn', False))
    layers = int(kwargs.get('layers', 1))
    rnntype = kwargs.get('rnntype', 'lstm')
    Seq2SeqModelType = Seq2SeqAttnModel if attn else Seq2SeqModel
    print(Seq2SeqModelType)
    model = Seq2SeqModelType(embedding1, embedding2, hsz, layers, rnntype)
    return model

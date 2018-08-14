import torch
import torch.nn as nn
import torch.nn.functional as F
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
        src_len = batch_dict['src_len']
        tgt = batch_dict['dst']

        dst = tgt[:, :-1]
        tgt = tgt[:, 1:]

        src_len, perm_idx = src_len.sort(0, descending=True)
        src = src[perm_idx]
        dst = dst[perm_idx]
        tgt = tgt[perm_idx]

        if self.gpu:
            src = src.cuda()
            dst = dst.cuda()
            tgt = tgt.cuda()
            src_len = src_len.cuda()

        return Variable(src), Variable(dst), Variable(src_len, requires_grad=False), Variable(tgt)

    # Input better be xch, x
    def forward(self, input):
        src = input[0]
        dst = input[1]
        src_len = input[2]
        rnn_enc_seq, final_encoder_state = self.encode(src, src_len)
        return self.decode(rnn_enc_seq, src_len, final_encoder_state, dst)

    def encode(self, src, src_len):
        src = src.transpose(0, 1).contiguous()
        embed_in_seq = self.embed_in(src)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embed_in_seq, src_len.data.tolist())
        output, hidden = self.encoder_rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        return output, hidden

    def input_i(self, embed_i, output_i):
        pass

    def bridge(self, final_encoder_state, context):
        pass

    def attn(self, output_t, context, src_mask=None):
        pass

    def decode_rnn(self, context, h_i, output_i, dst, src_mask):
        embed_out_seq = self.embed_out(dst)
        context_transpose = context.transpose(0, 1)
        outputs = []

        for i, embed_i in enumerate(embed_out_seq.split(1)):
            embed_i = self.input_i(embed_i, output_i)
            output_i, h_i = self.decoder_rnn(embed_i, h_i)
            output_i = self.attn(output_i, context_transpose, src_mask)
            output_i = self.dropout(output_i)
            outputs += [output_i]

        outputs = torch.stack(outputs)
        return outputs, h_i

    def decode(self, context, src_len, final_encoder_state, dst):

        src_mask = sequence_mask(src_len)
        if self.gpu:
            src_mask = src_mask.cuda()
        dst = dst.transpose(0, 1).contiguous()

        h_i, output_i = self.bridge(final_encoder_state, context)
        output, _ = self.decode_rnn(context, h_i, output_i, dst, src_mask)
        pred = self.prediction(output)
        return pred.transpose(0, 1).contiguous()

    def prediction(self, output):
        # Reform batch as (T x B, D)
        pred = self.probs(self.preds(output.view(output.size(0)*output.size(1),
                                                 -1)))
        # back to T x B x H -> B x T x H
        pred = pred.view(output.size(0), output.size(1), -1)
        return pred

    # B x K x T and here T is a list
    def run(self, batch_dict, **kwargs):
        src = batch_dict['src']
        src_len = batch_dict['src_len']
        src = torch.from_numpy(src) if type(src) == np.ndarray else src
        if type(src_len) == int:
            src_len = np.array([src_len])
        src_len = torch.from_numpy(src_len) if type(src_len) == np.ndarray else src_len
        if torch.is_tensor(src):
            src = torch.autograd.Variable(src, requires_grad=False)
        if torch.is_tensor(src_len):
            src_len = torch.autograd.Variable(src_len, requires_grad=False)

        if self.gpu:
            src = src.cuda()
            src_len = src_len.cuda()
        batch = []
        for src_i, src_len_i in zip(src, src_len):
            src_len_i = src_len_i.unsqueeze(0)
            batch += [self.beam_decode(src_i.view(1, -1), src_len_i, kwargs.get('beam', 1))[0]]

        return batch

    def beam_decode(self, src, src_len, K):
        with torch.no_grad():

            T = src.size(1)
            context, h_i = self.encode(src, src_len)
            src_mask = sequence_mask(src_len)
            dst_vocab = self.get_dst_vocab()
            GO = dst_vocab['<GO>']
            EOS = dst_vocab['<EOS>']

            paths = [[GO] for _ in range(K)]
            # K
            scores = torch.FloatTensor([0. for _ in range(K)])
            if self.gpu:
                scores = scores.cuda()
                src_mask = src_mask.cuda()
            # TBH
            context = torch.autograd.Variable(context.data.repeat(1, K, 1))
            h_i = (torch.autograd.Variable(h_i[0].data.repeat(1, K, 1)),
                   torch.autograd.Variable(h_i[1].data.repeat(1, K, 1)))
            h_i, dec_out = self.bridge(h_i, context)

            for i in range(T):
                lst = [path[-1] for path in paths]
                dst = torch.LongTensor(lst).type(src.data.type())
                mask_eos = dst == EOS
                mask_pad = dst == 0
                dst = dst.view(1, K)
                var = torch.autograd.Variable(dst)
                dec_out, h_i = self.decode_rnn(context, h_i, dec_out, var, src_mask)
                # 1 x K x V
                wll = self.prediction(dec_out).data
                # Just mask wll against end data
                V = wll.size(-1)
                dec_out = dec_out.squeeze(0)  # get rid of T=t dimension
                # K x V
                wll = wll.squeeze(0)  # get rid of T=t dimension

                if i > 0:
                    expanded_history = scores.unsqueeze(1).expand_as(wll)
                    wll.masked_fill_(mask_eos | mask_pad, 0)
                    sll = wll + expanded_history
                else:
                    sll = wll[0]

                flat_sll = sll.view(-1)
                best, best_idx = flat_sll.squeeze().topk(K, 0)
                best_beams = best_idx / V
                best_idx = best_idx % V
                new_paths = []
                for j, beam_id in enumerate(best_beams):
                    new_paths.append(paths[beam_id] + [best_idx[j]])
                    scores[j] = best[j]

                # Copy the beam state of the winners
                for hc in h_i:  # iterate over h, c
                    old_beam_state = hc.clone()
                    for i, beam_id in enumerate(best_beams):
                        H = hc.size(2)
                        src_beam = old_beam_state.view(-1, K, H)[:, beam_id]
                        dst_beam = hc.view(-1, K, H)[:, i]
                        dst_beam.data.copy_(src_beam.data)
                paths = new_paths

            return [p[1:] for p in paths], scores


class Seq2SeqModel(Seq2SeqBase):

    def __init__(self, embeddings_in, embeddings_out, **kwargs):
        super(Seq2SeqModel, self).__init__(embeddings_in, embeddings_out)

        self.hsz = kwargs['hsz']
        nlayers = kwargs['layers']
        rnntype = kwargs['rnntype']
        pdrop = kwargs.get('dropout', 0.5)
        enc_hsz = self.hsz
        if rnntype == 'blstm':
            enc_hsz = enc_hsz // 2
        dsz = embeddings_in.dsz
        self.gpu = kwargs.get('gpu', True)
        self.dropout = nn.Dropout(pdrop)
        self.encoder_rnn = pytorch_rnn(dsz, enc_hsz, rnntype, nlayers, pdrop)
        self.preds = nn.Linear(self.hsz, self.nc)
        self.decoder_rnn = pytorch_rnn_cell(dsz, self.hsz, rnntype, nlayers, pdrop)
        self.probs = nn.LogSoftmax(dim=1)

    def input_i(self, embed_i, output_i):
        return embed_i.squeeze(0)

    def bridge(self, final_encoder_state, context):
        return final_encoder_state, None

    def attn(self, output_t, context, src_mask=None):
        return output_t


class BaseAttention(nn.Module):

    def __init__(self, hsz):
        super(BaseAttention, self).__init__()
        self.hsz = hsz

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
        pass


class LuongBaseAttention(BaseAttention):
    def __init__(self, hsz):
        super(LuongBaseAttention, self).__init__(hsz)
        self.softmax = nn.Softmax(dim=1)
        self.W_a = nn.Linear(self.hsz, self.hsz, bias=False)
        self.W_c = nn.Linear(2 * self.hsz, hsz, bias=False)

    def _update(self, a, query_t, values_bth):
        # a = B x T
        # Want to apply over context, scaled by a
        # (B x 1 x T) (B x T x H) = (B x 1 x H)
        a = a.view(a.size(0), 1, a.size(1))
        c_t = torch.bmm(a, values_bth).squeeze(1)
        attended = torch.cat([c_t, query_t], 1)
        attended = F.tanh(self.W_c(attended))
        return attended


class LuongDotProductAttention(LuongBaseAttention):

    def __init__(self, hsz):
        super(LuongDotProductAttention, self).__init__(hsz)

    def _attention(self, query_t, keys_bth, keys_mask):
        a = torch.bmm(keys_bth, query_t.unsqueeze(2))
        a = a.squeeze(2).masked_fill(keys_mask == 0, -1e9)
        a = self.softmax(a)
        return a


class LuongGeneralAttention(LuongBaseAttention):

    def __init__(self, hsz):
        super(LuongGeneralAttention, self).__init__(hsz)

    def _attention(self, query_t, keys_bth, keys_mask):
        a = torch.bmm(keys_bth, self.W_a(query_t).unsqueeze(2))
        a = a.squeeze(2).masked_fill(keys_mask == 0, -1e9)
        a = self.softmax(a)
        return a


class Seq2SeqAttnModel(Seq2SeqBase):

    def __init__(self, embeddings_in, embeddings_out, **kwargs):
        super(Seq2SeqAttnModel, self).__init__(embeddings_in, embeddings_out)
        self.hsz = kwargs['hsz']
        nlayers = kwargs['layers']
        rnntype = kwargs['rnntype']
        pdrop = kwargs.get('dropout', 0.5)
        enc_hsz = self.hsz
        if rnntype == 'blstm':
            enc_hsz = enc_hsz // 2
        dsz = embeddings_in.dsz
        self.gpu = kwargs.get('gpu', True)
        self.encoder_rnn = pytorch_rnn(dsz, enc_hsz, rnntype, nlayers, pdrop)
        self.dropout = nn.Dropout(pdrop)
        self.decoder_rnn = pytorch_rnn_cell(self.hsz + dsz, self.hsz, rnntype, nlayers, pdrop)
        self.preds = nn.Linear(self.hsz, self.nc)
        self.probs = nn.LogSoftmax(dim=1)
        self.nlayers = nlayers
        attn_type = kwargs.get('attn_type', 'luong').lower()
        if attn_type == 'dot':
            self.attn_module = LuongDotProductAttention(enc_hsz)
        else:
            self.attn_module = LuongGeneralAttention(enc_hsz)

    def attn(self, output_t, context, src_mask=None):
        return self.attn_module(output_t, context, context, src_mask)

    def bridge(self, final_encoder_state, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.hsz)
        context_zeros = Variable(context.data.new(*h_size).zero_(), requires_grad=False)
        if type(final_encoder_state) is tuple:
            s1, s2 = final_encoder_state
            return (s1, s2), context_zeros
        else:
            return final_encoder_state, context_zeros

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

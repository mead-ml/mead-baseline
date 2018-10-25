from baseline.pytorch import (pytorch_rnn_cell,
                              pytorch_linear,
                              LuongDotProductAttention,
                              BahdanauAttention,
                              ScaledDotProductAttention,
                              LuongGeneralAttention)
from baseline.utils import Offsets
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from baseline.pytorch.transformer import subsequent_mask, TransformerDecoderStack


class RNNDecoder(torch.nn.Module):

    def __init__(self, tgt_embeddings, **kwargs):
        """Construct an RNN decoder.  It provides the input size, the rest is up to the impl.

        The default implementation provides an RNN cell, followed by a linear projection, out to a softmax

        :param input_dim: The input size
        :param kwargs:
        :return: void
        """
        super(RNNDecoder, self).__init__()
        self.hsz = kwargs['hsz']
        self.tgt_embeddings = tgt_embeddings
        rnntype = kwargs['rnntype']
        layers = kwargs['layers']
        feed_input = kwargs.get('feed_input', True)
        dsz = tgt_embeddings.get_dsz()
        if feed_input:
            self.input_i = self._feed_input
            dsz += self.hsz
        else:
            self.input_i = self._basic_input
        pdrop = kwargs.get('dropout', 0.5)
        self.decoder_rnn = pytorch_rnn_cell(dsz, self.hsz, rnntype, layers, pdrop)
        self.dropout = torch.nn.Dropout(pdrop)
        self.init_attn(**kwargs)
        self.preds = pytorch_linear(self.hsz, self.tgt_embeddings.get_vsz())

    @staticmethod
    def _basic_input(dst_embed_i, _):
        """
        In this function the destination embedding is passed directly to into the decoder.  The output of previous H
        is ignored.  This is implemented using a bound method to a field in the class for speed so that this decision
        is handled at initialization, not as a conditional in the training or inference

        :param embed_i: The embedding at i
        :param _: Ignored
        :return: basic input
        """
        return dst_embed_i.squeeze(0)

    @staticmethod
    def _feed_input(embed_i, attn_output_i):
        """
        In this function the destination embedding is concatenated with the previous attentional output and
        passed to the decoder. This is implemented using a bound method to a field in the class for speed
        so that this decision is handled at initialization, not as a conditional in the training or inference

        :param embed_i: The embedding at i
        :param output_i: This is the last H state
        :return: an input that is a concatenation of previous state and destination embedding
        """
        embed_i = embed_i.squeeze(0)
        return torch.cat([embed_i, attn_output_i], 1)

    def forward(self, encoder_outputs, embed_out_tbh):
        context_tbh = encoder_outputs.output
        src_mask = encoder_outputs.src_mask
        final_encoder_state = encoder_outputs.hidden
        h_i, output_i = self.bridge(final_encoder_state, context_tbh)
        output, _ = self.decode_rnn(context_tbh, h_i, output_i, embed_out_tbh, src_mask)
        pred = self.output(output)
        return pred.transpose(0, 1).contiguous()

    def decode_rnn(self, context_tbh, h_i, output_i, dst, src_mask):
        embed_out_tbh = self.tgt_embeddings(dst)

        context_bth = context_tbh.transpose(0, 1)
        outputs = []

        for i, embed_i in enumerate(embed_out_tbh.split(1)):
            # Input feeding would use previous attentional output in addition to destination embeddings
            embed_i = self.input_i(embed_i, output_i)
            output_i, h_i = self.decoder_rnn(embed_i, h_i)
            output_i = self.attn(output_i, context_bth, src_mask)
            output_i = self.dropout(output_i)
            # Attentional outputs
            outputs.append(output_i)

        outputs = torch.stack(outputs)
        return outputs, h_i

    def bridge(self, final_encoder_state, context):
        return final_encoder_state, None

    def attn(self, output_t, context, src_mask=None):
        return output_t

    def init_attn(self, **kwargs):
        pass

    def output(self, x):
        pred = F.log_softmax(self.preds(x.view(x.size(0)*x.size(1), -1)))
        pred = pred.view(x.size(0), x.size(1), -1)
        return pred

    def predict_one(self, src, encoder_outputs, **kwargs):
        K = kwargs.get('beam', 1)
        mxlen = kwargs.get('mxlen', 100)
        with torch.no_grad():
            context = encoder_outputs.output
            h_i = encoder_outputs.hidden
            src_mask = encoder_outputs.src_mask

            paths = [[Offsets.GO] for _ in range(K)]
            # K
            scores = torch.FloatTensor([0. for _ in range(K)]).type_as(context)
            # TBH
            context = torch.autograd.Variable(context.data.repeat(1, K, 1))
            h_i = (torch.autograd.Variable(h_i[0].data.repeat(1, K, 1)),
                   torch.autograd.Variable(h_i[1].data.repeat(1, K, 1)))
            h_i, dec_out = self.bridge(h_i, context)

            for i in range(mxlen):
                lst = [path[-1] for path in paths]
                dst = torch.LongTensor(lst).type(src.data.type())
                mask_eos = dst == Offsets.EOS
                mask_pad = dst == 0
                dst = dst.view(1, K)
                var = torch.autograd.Variable(dst)
                dec_out, h_i = self.decode_rnn(context, h_i, dec_out, var, src_mask)
                # 1 x K x V
                wll = self.output(dec_out).data
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


class RNNDecoderWithAttn(RNNDecoder):

    def __init__(self, tgt_embeddings, **kwargs):
        super(RNNDecoderWithAttn, self).__init__(tgt_embeddings, **kwargs)

    def init_attn(self, **kwargs):
        attn_type = kwargs.get('attn_type', 'bahdanau').lower()
        if attn_type == 'dot':
            self.attn_module = LuongDotProductAttention(self.hsz)
        elif attn_type == 'concat' or attn_type == 'bahdanau':
            self.attn_module = BahdanauAttention(self.hsz)
        elif attn_type == 'sdp':
            self.attn_module = ScaledDotProductAttention(self.hsz)
        else:
            self.attn_module = LuongGeneralAttention(self.hsz)

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


class TransformerDecoderWrapper(torch.nn.Module):

    def __init__(self, tgt_embeddings, **kwargs):
        super(TransformerDecoderWrapper, self).__init__()
        pdrop = float(kwargs.get('dropout', 0.5))
        layers = kwargs.get('layers', 1)
        d_model = int(kwargs.get('d_model', kwargs.get('hsz')))
        num_heads = kwargs.get('num_heads', 4)
        self.tgt_embeddings = tgt_embeddings
        self.transformer_decoder = TransformerDecoderStack(num_heads, d_model=d_model, pdrop=pdrop, scale=True, layers=layers)
        out_dsz = self.tgt_embeddings.get_dsz()
        self.proj = pytorch_linear(d_model, out_dsz) if d_model != out_dsz else lambda x: x
        self.preds = pytorch_linear(out_dsz, self.tgt_embeddings.get_vsz())

    def forward(self, encoder_output, dst):
        embed_out_bth = self.tgt_embeddings(dst)
        context_bth = encoder_output.output
        T = embed_out_bth.shape[1]
        dst_mask = subsequent_mask(T).type_as(embed_out_bth)
        src_mask = encoder_output.src_mask.unsqueeze(1).unsqueeze(1)
        output = self.transformer_decoder(embed_out_bth, context_bth, src_mask, dst_mask)
        output = self.proj(output)
        prob = self.output(output)
        return prob

    def predict_one(self, src, encoder_outputs, **kwargs):
        mxlen = kwargs.get('mxlen', 100)
        with torch.no_grad():
            # A single y value of <GO> to start
            ys = torch.ones(1, 1).fill_(Offsets.GO).type_as(src.data)

            for i in range(mxlen-1):
                # Make a mask of length T
                prob = self(encoder_outputs, ys)[:, -1]
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.data[0]
                # Add the word on to the end
                ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
                if next_word == Offsets.EOS:
                    break
        return ys, None

    def output(self, x):
        pred = F.log_softmax(self.preds(x.view(x.size(0)*x.size(1), -1)))
        pred = pred.view(x.size(0), x.size(1), -1)
        return pred

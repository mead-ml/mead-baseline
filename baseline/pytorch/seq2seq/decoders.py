import math
from functools import partial
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from baseline.utils import Offsets, exporter
from eight_mile.pytorch.layers import repeat_batch, gnmt_length_penalty, BeamSearchBase, rnn_cell, WeightTieDense
from baseline.pytorch.transformer import subsequent_mask, TransformerDecoderStack
from baseline.model import register_arc_policy, register_decoder, create_seq2seq_arc_policy
from baseline.pytorch.seq2seq.encoders import TransformerEncoderOutput
from baseline.pytorch.torchy import (
    tie_weight,
    pytorch_linear,
    LuongDotProductAttention,
    BahdanauAttention,
    ScaledDotProductAttention,
    LuongGeneralAttention,
)

__all__ = []
export = exporter(__all__)


class ArcPolicy(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, encoder_outputs, hsz, beam_width=1):
        pass


class AbstractArcPolicy(ArcPolicy):

    def __init__(self):
        super().__init__()

    def get_state(self, encoder_outputs):
        pass

    def forward(self, encoder_output, hsz, beam_width=1):
        h_i = self.get_state(encoder_output)
        context = encoder_output.output
        if beam_width > 1:
            with torch.no_grad():
                context = repeat_batch(context, beam_width)
                if type(h_i) is tuple:
                    h_i = repeat_batch(h_i[0], beam_width, dim=1), repeat_batch(h_i[1], beam_width, dim=1)
                else:
                    h_i = repeat_batch(h_i, beam_width, dim=1)
        batch_size = context.shape[0]
        h_size = (batch_size, hsz)
        with torch.no_grad():
            init_zeros = context.data.new(*h_size).zero_()
        return h_i, init_zeros, context


@register_arc_policy(name='default')
class TransferLastHiddenPolicy(AbstractArcPolicy):

    def __init__(self):
        super().__init__()

    def get_state(self, encoder_outputs):
        return encoder_outputs.hidden


@register_arc_policy(name='no_arc')
class NoArcPolicy(AbstractArcPolicy):

    def __init__(self):
        super().__init__()

    def get_state(self, encoder_outputs):
        final_encoder_state = encoder_outputs.hidden
        if isinstance(final_encoder_state, tuple):
            s1, s2 = final_encoder_state
            return s1 * 0, s2 * 0
        return final_encoder_state * 0


@register_decoder(name='vanilla')
class RNNDecoder(torch.nn.Module):

    def __init__(self, tgt_embeddings, **kwargs):
        """Construct an RNN decoder.  It provides the input size, the rest is up to the impl.

        The default implementation provides an RNN cell, followed by a linear projection, out to a softmax

        :param input_dim: The input size
        :param kwargs:
        :return: void
        """
        super().__init__()
        self.hsz = kwargs['hsz']
        self.arc_policy = create_seq2seq_arc_policy(**kwargs)
        self.tgt_embeddings = tgt_embeddings
        rnntype = kwargs.get('rnntype', 'lstm')
        layers = kwargs.get('layers', 1)
        feed_input = kwargs.get('feed_input', True)
        dsz = tgt_embeddings.get_dsz()
        if feed_input:
            self.input_i = self._feed_input
            dsz += self.hsz
        else:
            self.input_i = self._basic_input
        pdrop = kwargs.get('dropout', 0.5)
        self.decoder_rnn = rnn_cell(dsz, self.hsz, rnntype, layers, pdrop)
        self.dropout = torch.nn.Dropout(pdrop)
        self.init_attn(**kwargs)

        do_weight_tying = bool(kwargs.get('tie_weights', True))

        if do_weight_tying:
            if self.hsz != self.tgt_embeddings.get_dsz():
                raise ValueError("weight tying requires hsz == embedding dsz, got {} hsz and {} dsz".format(self.hsz, self.tgt_embeddings.get_dsz()))
            self.preds = WeightTieDense(self.tgt_embeddings)
        else:
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

    def forward(self, encoder_outputs, dst):
        src_mask = encoder_outputs.src_mask
        h_i, output_i, context_bth = self.arc_policy(encoder_outputs, self.hsz)
        output_tbh, _ = self.decode_rnn(context_bth, h_i, output_i, dst.transpose(0, 1), src_mask)
        pred = self.output(output_tbh)
        return pred.transpose(0, 1).contiguous()

    def decode_rnn(self, context_bth, h_i, output_i, dst_bth, src_mask):
        embed_out_bth = self.tgt_embeddings(dst_bth)

        outputs = []

        for i, embed_i in enumerate(embed_out_bth.split(1)):
            # Input feeding would use previous attentional output in addition to destination embeddings
            embed_i = self.input_i(embed_i, output_i)
            output_i, h_i = self.decoder_rnn(embed_i, h_i)
            output_i = self.attn(output_i, context_bth, src_mask)
            output_i = self.dropout(output_i)
            # Attentional outputs
            outputs.append(output_i)

        outputs_tbh = torch.stack(outputs)
        return outputs_tbh, h_i

    def attn(self, output_t, context, src_mask=None):
        return output_t

    def init_attn(self, **kwargs):
        pass

    def output(self, x):
        pred = F.log_softmax(self.preds(x.view(x.size(0)*x.size(1), -1)), dim=-1)
        pred = pred.view(x.size(0), x.size(1), -1)
        return pred

    class BeamSearch(BeamSearchBase):

        def __init__(self, parent, **kwargs):
            super().__init__(**kwargs)
            self.parent = parent

        def init(self, encoder_outputs):
            """Tile batches for encoder inputs and the likes."""
            src_mask = repeat_batch(encoder_outputs.src_mask, self.K)
            h_i, dec_out, context = self.parent.arc_policy(encoder_outputs, self.parent.hsz, self.K)
            return h_i, dec_out, context, src_mask

        def step(self, paths, extra):
            """Calculate the probs of the next output and update state."""
            h_i, dec_out, context, src_mask = extra
            last = paths[:, :, -1].view(1, -1)
            dec_out, h_i = self.parent.decode_rnn(context, h_i, dec_out, last, src_mask)
            probs = self.parent.output(dec_out)
            dec_out = dec_out.squeeze(0)
            return probs, (h_i, dec_out, context, src_mask)

        def update(self, beams, extra):
            """Select the correct hidden states and outputs to used based on the best performing beams."""
            h_i, dec_out, context, src_mask = extra
            h_i = tuple(hc[:, beams, :] for hc in h_i)
            dec_out = dec_out[beams, :]
            return h_i, dec_out, context, src_mask

    def beam_search(self, encoder_outputs, **kwargs):
        alpha = kwargs.get('alpha')
        if alpha is not None:
            kwargs['length_penalty'] = partial(gnmt_length_penalty, alpha=alpha)
        return RNNDecoder.BeamSearch(parent=self, **kwargs)(encoder_outputs)


@register_decoder(name='default')
class RNNDecoderWithAttn(RNNDecoder):

    def __init__(self, tgt_embeddings, **kwargs):
        super().__init__(tgt_embeddings, **kwargs)

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


@register_decoder(name='transformer')
class TransformerDecoderWrapper(torch.nn.Module):

    def __init__(self, tgt_embeddings, dropout=0.5, layers=1, hsz=None, num_heads=4, scale=True, **kwargs):
        super().__init__()
        self.tgt_embeddings = tgt_embeddings
        dsz = self.tgt_embeddings.get_dsz()
        if hsz is None:
            hsz = dsz

        d_ff = int(kwargs.get('d_ff', 4 * hsz))
        rpr_k = kwargs.get('rpr_k')
        d_k = kwargs.get('d_k')
        activation = kwargs.get('activation', 'relu')
        scale = bool(kwargs.get('scale', True))

        self.transformer_decoder = TransformerDecoderStack(num_heads, d_model=hsz, d_ff=d_ff,
                                                           pdrop=dropout, scale=scale,
                                                           layers=layers, rpr_k=rpr_k, d_k=d_k, activation_type=activation)

        self.proj_to_dsz = self._identity
        self.proj_to_hsz = self._identity
        if hsz != dsz:
            self.proj_to_hsz = pytorch_linear(dsz, hsz)
            self.proj_to_dsz = pytorch_linear(hsz, dsz)
            del self.proj_to_dsz.weight
            self.proj_to_dsz.weight = torch.nn.Parameter(self.proj_to_hsz.weight.transpose(0, 1), requires_grad=True)

        do_weight_tying = bool(kwargs.get('tie_weights', True))
        if do_weight_tying:
            if hsz != self.tgt_embeddings.get_dsz():
                raise ValueError("weight tying requires hsz == embedding dsz, got {} hsz and {} dsz".format(self.hsz, self.tgt_embeddings.get_dsz()))
            self.preds = WeightTieDense(self.tgt_embeddings)
        else:
            self.preds = pytorch_linear(dsz, self.tgt_embeddings.get_vsz())

    def _identity(self, x):
        return x

    def forward(self, encoder_output, dst):
        embed_out_bth = self.tgt_embeddings(dst)
        embed_out_bth = self.proj_to_hsz(embed_out_bth)
        context_bth = encoder_output.output
        T = embed_out_bth.shape[1]
        dst_mask = subsequent_mask(T).type_as(embed_out_bth)
        src_mask = encoder_output.src_mask.unsqueeze(1).unsqueeze(1)
        output = self.transformer_decoder((embed_out_bth, context_bth, src_mask, dst_mask))
        output = self.proj_to_dsz(output)
        prob = self.output(output)
        return prob

    def output(self, x):
        pred = F.log_softmax(self.preds(x.view(x.size(0)*x.size(1), -1)), dim=-1)
        pred = pred.view(x.size(0), x.size(1), -1)
        return pred

    class BeamSearch(BeamSearchBase):

        def __init__(self, parent, **kwargs):
            super().__init__(**kwargs)
            self.parent = parent

        def init(self, encoder_outputs):
            """Tile for the batch of the encoder inputs."""
            encoder_outputs = TransformerEncoderOutput(
                repeat_batch(encoder_outputs.output, self.K),
                repeat_batch(encoder_outputs.src_mask, self.K)
            )
            return encoder_outputs

        def step(self, paths, extra):
            """Calculate the probs for the last item based on the full path."""
            B, K, T = paths.size()
            assert K == self.K
            return self.parent(extra, paths.view(B * K, T))[:, -1], extra

        def update(self, beams, extra):
            """There is no state for the transformer so just pass it."""
            return extra

    def beam_search(self, encoder_outputs, **kwargs):
        return TransformerDecoderWrapper.BeamSearch(parent=self, **kwargs)(encoder_outputs)

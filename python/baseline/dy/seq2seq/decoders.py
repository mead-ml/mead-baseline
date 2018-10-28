import dynet as dy
from baseline.utils import Offsets
from baseline.dy.dynety import DynetModel, Linear, Attention
from baseline.dy.transformer import subsequent_mask, TransformerDecoderStack
from baseline.model import (
    register_decoder,
    register_arc_policy,
    create_seq2seq_arc_policy
)


class ArcPolicy(object):
    def __init__(self):
        super(ArcPolicy, self).__init__()

    def get_state(self, encoder_outputs):
        pass

    def __call__(self, encoder_output, hsz, beam_width=1):
        h_i = self.get_state(encoder_output)
        context = encoder_output.output
        _, batchsz = h_i[0].dim()
        init_zeros = dy.zeros((hsz,), batch_size=batchsz)
        return h_i, init_zeros, context


@register_arc_policy(name='default')
class TransferLastHiddenPolicy(ArcPolicy):
    def __init__(self):
        super(TransferLastHiddenPolicy, self).__init__()

    def get_state(self, encoder_outputs):
        print(encoder_outputs.hidden.dim())
        return encoder_outputs.hidden


@register_arc_policy(name='no_arc')
class NoArcPolicy(ArcPolicy):
    def __init__(self):
        super(NoArchPolicy, self).__init__()

    def get_state(self, encoder_outputs):
        final_state = encoder_outputs.hidden
        return [x * 0 for x in final_state]


class DecoderBase(DynetModel):
    def __init__(self, pc):
        super(DecoderBase, self).__init__(pc)

    def __call__(self, encoder_output, dst):
        pass

    def predict_one(self, src, encoder_outputs, **kwargs):
        pass


@register_decoder(name='vanilla')
class RNNDecoder(DecoderBase):
    def __init__(self, tgt_embeddings, **kwargs):
        pc = kwargs.pop('pc').add_subcollection(name=kwargs.get('name', 'rnn-decoder'))
        super(RNNDecoder, self).__init__(pc)
        self.hsz = kwargs['hsz']
        self.arc_policy = create_seq2seq_arc_policy(**kwargs)
        self.tgt_embeddings = tgt_embeddings
        rnntype = kwargs.get('rnntype', 'lstm')
        layers = kwargs['layers']
        feed_input = kwargs.get('feed_input', True)
        dsz = tgt_embeddings.get_dsz()
        if feed_input:
            self.input_i = self._feed_input
            dsz += self.hsz
        else:
            self.input_i = self._basic_input
        self.pdrop = kwargs.get('dropout', 0.5)
        self.decoder_rnn = dy.VanillaLSTMBuilder(layers, dsz, self.hsz, self.pc)
        self.init_attn(**kwargs)
        self.preds = Linear(self.tgt_embeddings.get_vsz(), self.hsz, self.pc)

    @staticmethod
    def _basic_input(dst_embed_i, _):
        return dst_embed_i

    @staticmethod
    def _feed_input(dst_embed_i, attn_output_i):
        return dy.concatenate([dst_embed_i, attn_output_i])

    def init_attn(self, **kwargs):
        pass

    def _attn_first(self, context):
        pass

    def attn(self, output_t, context, src_mask):
        return output_t

    def __call__(self, encoder_outputs, dst, train=False):
        src_mask = encoder_outputs.src_mask
        h_i, output_i, context = self.arc_policy(encoder_outputs, self.hsz)
        output = self.decode_rnn(context, h_i, output_i, dst, train, src_mask)
        return self.output(output)

    def decode_rnn(self, context, h_i, output_i, dst, train, src_mask):
        embed_out = self.tgt_embeddings.encode(dst, train)
        outputs = []
        self._attn_first(context)
        rnn_state = self.decoder_rnn.initial_state(h_i)
        for embed_i in embed_out:
            embed_i = self.input_i(embed_i, output_i)
            rnn_state = rnn_state.add_input(embed_i)
            rnn_output_i = rnn_state.output()
            output_i = self.attn(rnn_output_i, None, src_mask)
            outputs.append(output_i)
        return outputs

    def output(self, x):
        return [dy.log_softmax(self.preds(y)) for y in x]

@register_decoder(name='default')
class RNNDecoderWithAttn(RNNDecoder):
    def __init__(self, tgt_embeddings, **kwargs):
        super(RNNDecoderWithAttn, self).__init__(tgt_embeddings, **kwargs)

    def init_attn(self, **kwargs):
        self.attn_module = Attention(self.hsz, self.pc)

    def _attn_first(self, context):
        context_mx = dy.concatenate_cols(context)
        self._attn = self.attn_module(context_mx)

    def attn(self, output_t, context, src_mask=None):
        return self._attn(output_t, src_mask)

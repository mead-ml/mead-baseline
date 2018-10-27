from collections import namedtuple
import dynet as dy
from baseline.utils import sequence_mask
from baseline.model import register_encoder
from baseline.dy.transformer import TransformerEncoderStack
from baseline.dy.dynety import DynetModel, Linear, rnn_forward_with_state


RNNEncoderOutput = namedtuple("RNNEncoderOutput", ("output", "hidden", "src_mask"))
TransformerEncoderOutput = namedtuple("TransformerEncoderOutput", ("output", "src_mask"))

class EncoderBase(object):
    def __init__(self, *args, **kwargs):
        super(EncoderBase, self).__init__(*args, **kwargs)

    def encode(self, embed_in, src_len, **kwargs):
        pass

def _make_sequence_mask(lengths, max_len):
    return dy.inputTensor(sequence_mask(lengths, max_len))


@register_encoder(name='default')
class RNNEncoder(EncoderBase, DynetModel):
    def __init__(self, insz, pc, hsz=None, rnntype='blstm', layers=1, pdrop=0.5, residual=False, create_src_mask=True, name='rnn-encoder', **kwargs):
        pc = pc.add_subcollection(name=name)
        super(RNNEncoder, self).__init__(pc)
        self.residual = residual
        hidden = hsz if hsz is not None else insz
        if rnntype == 'blstm':
            self.lstm_forward = dy.VanillaLSTMBuilder(layers, insz, hidden // 2, self.pc)
            self.lstm_backward = dy.VanillaLSTMBuilder(layers, insz, hidden // 2, self.pc)
        else:
            self.lstm_forward = dy.VanillaLSTMBuilder(layers, insz, hidden, self.pc)
            self.lstm_backward = None
        self.src_mask_fn = _make_sequence_mask if create_src_mask else lambda x, y: None

    def encode(self, embed_in, src_len, train=False, **kwargs):
        """Input Shape: ((T, H), B). Output Shape: [((H,), B)] * T"""
        embed_in = list(embed_in)
        forward, forward_state = rnn_forward_with_state(self.lstm_forward, embed_in, src_len)
        # TODO: add dropout
        if self.lstm_backward is not None:
            backward, backward_state = rnn_forward_with_state(self.lstm_backward, embed_in)
            output = [dy.concatenate([f, b]) for f, b in zip(forward, backward)]
            hidden = [dy.concatenate([f, b]) for f, b in zip(forward, backward)]
        else:
            output = forward
            hidden = forward_state
        return RNNEncoderOutput(
            output=[o + e for o, e in zip(output, embed_in)] if self.residual else output,
            hidden=hidden,
            src_mask=self.src_mask_fn(src_len, len(output))
        )


@register_encoder(name='transformer')
class TransformerEncoderWrapper(EncoderBase, DynetModel):
    def __init__(self, dsz, pc, hsz=None, num_heads=4, layers=1, dropout=0.5, name='transformer-encoder-wrapper', **kwargs):
        pc = pc.add_subcollection(name=name)
        super(TransformerEncoderWrapper, self).__init__(pc)
        if hsz is None:
            hsz = dsz
        self.proj = Linear(dsz, hsz, pc) if hsz != dsz else lambda x: x
        self.transformer = TransformerEncoderStack(num_heads, d_model=hsz, pc=pc, pdrop=dropout, scale=True, layers=layers)

    def encode(self, embed_in, src_len, train=False, **kwargs):
        """Input shape: ((T, H), B) Output Shape: [((H,), B)] * T"""
        T = embed_in.dim()[0][0]
        embed_in = dy.transpose(embed_in)
        src_mask = _make_sequence_mask(src_len, T)
        x = self.proj(embed_in)
        output = self.transformer(x, src_mask, train=train)
        print(output.dim())
        return TransformerEncoderOutput(output=output, src_mask=src_mask)

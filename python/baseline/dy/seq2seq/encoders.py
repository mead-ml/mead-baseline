from collections import namedtuple
import dynet as dy
from baseline.model import register_encoder
from baseline.dy.transformer import TransformerEncoderStack
from baseline.dy.dynety import DynetModel, Linear, rnn_forward_with_state, sequence_mask, unsqueeze


RNNEncoderOutput = namedtuple("RNNEncoderOutput", ("output", "hidden", "src_mask"))
TransformerEncoderOutput = namedtuple("TransformerEncoderOutput", ("output", "src_mask"))

class EncoderBase(DynetModel):
    def __init__(self, pc):
        super(EncoderBase, self).__init__(pc)

    def encode(self, embed_in, src_len, **kwargs):
        pass


@register_encoder(name='default')
class RNNEncoder(EncoderBase):
    def __init__(self, dsz, pc, hsz=None, rnntype='blstm', layers=1, pdrop=0.5, residual=False, create_src_mask=True, name='rnn-encoder', **kwargs):
        pc = pc.add_subcollection(name=name)
        super(RNNEncoder, self).__init__(pc)
        self.residual = residual
        hidden = hsz if hsz is not None else dsz
        if rnntype == 'blstm':
            self.lstm_forward = dy.VanillaLSTMBuilder(layers, dsz, hidden // 2, self.pc)
            self.lstm_backward = dy.VanillaLSTMBuilder(layers, dsz, hidden // 2, self.pc)
        else:
            self.lstm_forward = dy.VanillaLSTMBuilder(layers, dsz, hidden, self.pc)
            self.lstm_backward = None
        self.src_mask_fn = sequence_mask if create_src_mask else lambda x, y: (None, None)
        self.pdrop = pdrop

    def dropout(self, train):
        if train:
            self.lstm_forward.set_dropout(self.pdrop)
            if self.lstm_backward is not None:
                self.lstm_forward.set_dropout(self.pdrop)
        else:
            self.lstm_forward.disable_dropout()
            if self.lstm_backward is not None:
                self.lstm_forward.disable_dropout()

    def __call__(self, embed_in, src_len, train=False, **kwargs):
        """Input Shape: ((T, H), B). Output Shape: [((H,), B)] * T"""
        embed_in = list(embed_in)
        self.dropout(train)
        forward, forward_state = rnn_forward_with_state(self.lstm_forward, embed_in, src_len)
        if self.lstm_backward is not None:

            backward, backward_state = rnn_forward_with_state(self.lstm_backward, embed_in)
            output = [dy.concatenate([f, b]) for f, b in zip(forward, backward)]
            hidden = [dy.concatenate([f, b]) for f, b in zip(forward_state, backward_state)]
        else:
            output = forward
            hidden = forward_state
        return RNNEncoderOutput(
            output=[o + e for o, e in zip(output, embed_in)] if self.residual else output,
            hidden=hidden,
            src_mask=self.src_mask_fn(src_len, len(output))
        )


@register_encoder(name='transformer')
class TransformerEncoderWrapper(EncoderBase):
    def __init__(self, dsz, pc, hsz=None, num_heads=4, layers=1, dropout=0.5, name='transformer-encoder-wrapper', **kwargs):
        pc = pc.add_subcollection(name=name)
        super(TransformerEncoderWrapper, self).__init__(pc)
        if hsz is None:
            hsz = dsz
        self.proj = Linear(hsz, dsz, pc) if hsz != dsz else lambda x: x
        self.transformer = TransformerEncoderStack(num_heads, d_model=hsz, pc=pc, pdrop=dropout, scale=True, layers=layers)

    def __call__(self, embed_in, src_len, train=False, **kwargs):
        """Input shape: ((T, H), B) Output Shape: [((H,), B)] * T"""
        T = embed_in.dim()[0][0]
        embed_in = dy.transpose(embed_in)
        src_mask = sequence_mask(src_len, T)
        src_mask = [unsqueeze(m, 2) for m in src_mask]
        x = self.proj(embed_in)
        output = self.transformer(x, src_mask, train=train)
        output = [out for out in dy.transpose(output)]
        return TransformerEncoderOutput(output=output, src_mask=src_mask)

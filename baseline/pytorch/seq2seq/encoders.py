from collections import namedtuple
from baseline.pytorch.torchy import sequence_mask, pytorch_linear, pytorch_lstm
from eight_mile.pytorch.layers import TransformerEncoderStack
from baseline.model import register_encoder
from eight_mile.pytorch.layers import *
import torch


RNNEncoderOutput = namedtuple("RNNEncoderOutput", ("output", "hidden", "src_mask"))


def _make_src_mask(output, lengths):
    T = output.shape[1]
    src_mask = sequence_mask(lengths, T).type_as(lengths.data).to(device=output.device)
    return src_mask


@register_encoder(name='default')
class RNNEncoder(torch.nn.Module):

    def __init__(self, dsz=None, hsz=None, rnntype='blstm', layers=1, pdrop=0.5, residual=False, create_src_mask=True, **kwargs):
        super().__init__()
        self.residual = residual
        hidden = hsz if hsz is not None else dsz
        Encoder = LSTMEncoderAll if rnntype == 'lstm' else BiLSTMEncoderAll
        self.rnn = Encoder(dsz, hidden, layers, pdrop, batch_first=True)
        self.src_mask_fn = _make_src_mask if create_src_mask is True else lambda x, y: None

    def forward(self, btc, lengths):
        output, hidden = self.rnn((btc, lengths))
        return RNNEncoderOutput(output=output + btc if self.residual else output,
                                hidden=hidden,
                                src_mask=self.src_mask_fn(output, lengths))


TransformerEncoderOutput = namedtuple("TransformerEncoderOutput", ("output", "src_mask"))


@register_encoder(name='transformer')
class TransformerEncoderWrapper(torch.nn.Module):

    def __init__(self, dsz, hsz=None, num_heads=4, layers=1, dropout=0.5,
                 activation='relu',
                 rpr_k=None,
                 layer_norm_eps=1e-6,
                 layer_drop=0.0, scale=True, rpr_value_on=True, ra_type=None,
                 d_k=None,
                 d_ff=None,
                 transformer_type=None,
                 **kwargs):
        super().__init__()
        if hsz is None:
            hsz = dsz
        self.proj = pytorch_linear(dsz, hsz) if hsz != dsz else self._identity
        if d_ff is None:
            d_ff = 4 * hsz

        self.transformer = TransformerEncoderStack(num_heads, d_model=hsz, d_ff=d_ff,
                                                   pdrop=dropout, scale=scale, layers=layers,
                                                   rpr_k=rpr_k, d_k=d_k, activation=activation, layer_drop=layer_drop,
                                                   layer_norm_eps=layer_norm_eps,
                                                   rpr_value_on=rpr_value_on, ra_type=ra_type, transformer_type=transformer_type)

    def _identity(self, x):
        return x

    def forward(self, bth, lengths):
        T = bth.shape[1]
        src_mask = sequence_mask(lengths, T).type_as(lengths.data).to(bth.device)
        bth = self.proj(bth)
        output = self.transformer((bth, src_mask.unsqueeze(1).unsqueeze(1)))
        return TransformerEncoderOutput(output=output, src_mask=src_mask)

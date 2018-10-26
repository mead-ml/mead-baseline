from collections import namedtuple
from baseline.pytorch.torchy import sequence_mask, pytorch_linear, pytorch_rnn
from baseline.pytorch.transformer import TransformerEncoderStack
import torch


class Orientation:
    BTH, TBH = range(0, 2)


RNNEncoderOutput = namedtuple("RNNEncoderOutput", ("output", "hidden", "src_mask", "orientation"))


def _make_src_mask(output, lengths):
    T = output.shape[0]
    src_mask = sequence_mask(lengths, T).type_as(lengths.data)
    return src_mask


class RNNEncoder(torch.nn.Module):

    def __init__(self, insz, hsz=None, rnntype='blstm', layers=1, pdrop=0.5, residual=False, create_src_mask=True, **kwargs):
        super(RNNEncoder, self).__init__()
        self.residual = residual
        hidden = hsz if hsz is not None else insz
        self.rnn = pytorch_rnn(insz, hidden, rnntype, layers, pdrop)
        self.src_mask_fn = _make_src_mask if create_src_mask is True else lambda x, y: None

    def forward(self, tbc, lengths):
        packed = torch.nn.utils.rnn.pack_padded_sequence(tbc, lengths.tolist())
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        return RNNEncoderOutput(output=output + tbc if self.residual else output,
                                hidden=hidden,
                                src_mask=self.src_mask_fn(output, lengths),
                                orientation=Orientation.TBH)


TransformerEncoderOutput = namedtuple("TransformerEncoderOutput", ("output", "src_mask"))


class TransformerEncoderWrapper(torch.nn.Module):

    def __init__(self, insz, hsz=None, num_heads=4, layers=1, dropout=0.5, **kwargs):
        super(TransformerEncoderWrapper, self).__init__()
        if hsz is None:
            hsz = insz
        self.proj = pytorch_linear(insz, hsz) if hsz != insz else self._identity
        self.transformer = TransformerEncoderStack(num_heads, d_model=hsz, pdrop=dropout, scale=True, layers=layers)

    def _identity(self, x):
        return x

    def forward(self, bth, lengths):
        T = bth.shape[1]
        src_mask = sequence_mask(lengths, T).type_as(lengths.data)
        bth = self.proj(bth)
        output = self.transformer(bth, src_mask.unsqueeze(1).unsqueeze(1))
        return TransformerEncoderOutput(output=output, src_mask=src_mask)

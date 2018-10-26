
__all__ = []
exporter = export(__all__)


@exporter
class Encoderbase(object):
    def __init__(self):
        pass

    def encode(self, embed_in, src_len, pkeep, **kwargs):
        pass


@register_encoder(name='default')
class RNNEncoder(EncoderBase, DynetModel):
    def __init__(self, insz, hsz=None, rnntype='blstm', layers=1, pdrop=0.5, residual=False, create_crf_mask=True, **kwargs):
        super(RNNEncoder, self).__init__()
        self.residual = residual
        hidden = hsz if hsz is not None else insz
        if self.rnntype == 'blstm':
            self.lstm_forward = dy.VanillaLSTMBuilder(layers, insz, hidden // 2, self.pc)
            self.lstm_backward = dy.VanillaLSTMBuilder(layers, insz, hidden // 2, self.pc)
        else:
            self.lstm_forward = dy.VanillaLSTMBuilder(layers, insz, hidden, self.pc)
            self.lstm_backward = None

    def encode(self, embed_in, src_len, train=False, **kwargs):
        pass


@register_encoder(name='transformer')
class TransformerEncoderWrapper(Encoderbase, DynetModel):
    def __init__(self):
        pass

    def encode(self, embed_in, src_len, train=False, **kwargs):
        pass

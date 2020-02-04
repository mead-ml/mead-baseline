import torch
import torch.nn as nn
import torch.nn.functional as F
from baseline.pytorch.classify import ClassifierModelBase
from baseline.model import register_model

def ngrams(x, filtsz, mxlen):
    chunks = []
    for i in range(mxlen - filtsz + 1):
        chunk = x[:, i:i+filtsz, :]
        chunks += [chunk]
    chunks = torch.stack(chunks, 1)
    return chunks


@register_model(task='classify', name='rnf')
class RNFWordClassifier(ClassifierModelBase):

    def __init__(self):
        super(RNFWordClassifier, self).__init__()

    def init_pool(self, dsz, **kwargs):
        self.filtsz = kwargs['filtsz']
        self.mxlen = kwargs.get('mxlen', 100)
        pdrop = kwargs.get('dropout', 0.4)
        self.dropout_pre = nn.Dropout(pdrop)
        rnnsz = kwargs.get('rnnsz', 300)
        self.rnf = nn.LSTM(dsz, rnnsz, batch_first=True)
        self.pool_dropout = nn.Dropout(kwargs.get('pool_dropout', 0.))
        return rnnsz

    def pool(self, btc, lengths):

        btc = self.dropout_pre(btc)
        btfc = ngrams(btc, self.filtsz, self.mxlen)
        B, T, F, C = btfc.shape
        btc = btfc.view(B*T, F, C)
        output, hidden = self.rnf(btc)
        hidden = hidden[0].view(hidden[0].shape[1:])
        btc = hidden.view(B, T, -1)
        bc = btc.max(1)[0]
        return self.pool_dropout(bc)


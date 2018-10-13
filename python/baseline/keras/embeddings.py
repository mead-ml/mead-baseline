from baseline.utils import write_json
from keras.layers import (Embedding,
                          Input)
import numpy as np
from baseline.embeddings import register_embeddings


class KerasEmbeddings(object):

    def __init__(self):
        super(KerasEmbeddings).__init__()

    def get_vsz(self):
        pass

    def get_dsz(self):
        pass

    def save_md(self, target):
        pass

    def encode(self, x):
        return self(x)

    @classmethod
    def create(cls, model, name, **kwargs):
        return cls(name, vsz=model.vsz, dsz=model.dsz, weights=model.weights, **kwargs)

@register_embeddings(name='default')
class LookupTableEmbeddings(KerasEmbeddings):

    def __init__(self, name, **kwargs):
        super(LookupTableEmbeddings, self).__init__()
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        self.mxlen = kwargs.get('mxlen', 100)
        self.name = name
        self.x = Input(shape=(self.mxlen,), dtype='int32', name=name)
        self.weights = kwargs.get('weights')
        if self.weights is None:
            self.weights = np.zeros((self.vsz, self.dsz))
        self.lut = Embedding(input_dim=self.vsz, output_dim=self.dsz, weights=[self.weights], input_length=self.mxlen, trainable=self.finetune)

    def get_dsz(self):
        return self.dsz

    def get_vsz(self):
        return self.vsz

    def save_md(self, target):
        write_json({'vsz': self.vsz, 'dsz': self.dsz}, target)

    def encode(self):
        return self.lut(self.x)

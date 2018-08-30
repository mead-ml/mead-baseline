from baseline.keras.classify import SequentialWordClassifierBase

from keras.layers import (Dropout,
                          GlobalMaxPooling1D,
                          TimeDistributed,
                          Lambda,
                          LSTM,
                          CuDNNLSTM)

import keras.backend as K


def make_ngram_fn(filtsz, mxlen):

    def make_ngram_dim(x):
        chunks = []
        # For each temporal lag...
        for i in range(mxlen - filtsz + 1):
            # Get an N-gram
            chunk = x[:, i:i+filtsz, :]
            chunk = K.expand_dims(chunk, 1)
            chunks.append(chunk)
        return K.concatenate(chunks, 1)
    return make_ngram_dim


# TODO: add docs, comments
class RNFWordClassifier(SequentialWordClassifierBase):

    def __init__(self):

        super(RNFWordClassifier, self).__init__()

    def _pool(self, dsz, **kwargs):
        filtsz = kwargs['filtsz']
        pdrop = kwargs.get('dropout', 0.5)
        rnnsz = kwargs['rnnsz']
        mxlen = kwargs.get('mxlen', 100)
        self.impl.add(Dropout(rate=pdrop))
        self.impl.add(Lambda(make_ngram_fn(filtsz, mxlen)))
        #self.impl.add(TimeDistributed(LSTM(rnnsz)))
        self.impl.add(TimeDistributed(CuDNNLSTM(rnnsz)))
        self.impl.add(GlobalMaxPooling1D())
        self.impl.add(Dropout(rate=pdrop))


def create_model(embeddings, labels, **kwargs):
    return RNFWordClassifier.create(embeddings, labels, **kwargs)


def load_model(name, **kwargs):
    return RNFWordClassifier.load(name, **kwargs)

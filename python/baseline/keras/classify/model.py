import keras.models
from keras.layers import (Dense,
                          Conv1D,
                          Embedding,
                          Input,
                          GlobalMaxPooling1D,
                          Dropout,
                          LSTM,
                          GlobalAveragePooling1D)

from keras.utils import np_utils
from baseline.utils import listify
from baseline.model import Classifier, load_classifier_model, create_classifier_model
import json


class WordClassifierBase(Classifier):

    def __init__(self):
        pass

    def save(self, basename):
        self.impl.save(basename, overwrite=True)

        with open(basename + '.labels', 'w') as f:
            json.dump(self.labels, f)

        with open(basename + '.vocab', 'w') as f:
            json.dump(self.vocab, f)

    def classify(self, batch_dict):
        batch_time = batch_dict['x']
        batchsz = batch_time.shape[0]
        return self.impl.fit(batch_time, batch_size=batchsz)

    def classify(self, batch_time):

        batchsz = batch_time.shape[0]
        probs = self.impl.predict(batch_time, batchsz)

        results = []
        for b in range(batchsz):
            outcomes = [(self.labels[id_i], prob_i) for id_i, prob_i in enumerate(probs[b])]
            results.append(outcomes)
        return results

    def make_input(self, batch_dict):
        x = batch_dict['x']
        y = np_utils.to_categorical(batch_dict['y'], len(self.labels))
        return x, y

    def get_labels(self):
        return self.labels

    def get_vocab(self):
        return self.vocab


class GraphWordClassifierBase(WordClassifierBase):

    def __init__(self):
        super(GraphWordClassifierBase, self).__init__()

    @classmethod
    def create(cls, embeddings, labels, **kwargs):
        w2v = embeddings['word']
        model = cls()
        model.labels = labels
        model.vocab = w2v.vocab
        mxlen = int(kwargs.get('mxlen', 100))
        finetune = bool(kwargs.get('finetune', True))
        nc = len(labels)
        x = Input(shape=(mxlen,), dtype='int32', name='input')

        vocab_size = w2v.weights.shape[0]
        embedding_dim = w2v.dsz

        lut = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[w2v.weights], input_length=mxlen, trainable=finetune)

        embed = lut(x)

        last_layer, input_dim = model._pool(embed, **kwargs)
        last_layer = model._stacked(last_layer, **kwargs)

        dense = Dense(units=nc, input_dim=input_dim, activation='softmax')(last_layer)
        model.impl = keras.models.Model(inputs=[x], outputs=[dense])
        model.impl.summary()
        return model

    def _stacked(self, pooled, **kwargs):
        pass

    def _pool(self, embed, **kwargs):
        pass

    @classmethod
    def load(cls, basename, **kwargs):
        model = cls()

        model.impl = keras.models.load_model(basename, **kwargs)
        with open(basename + '.labels', 'r') as f:
            model.labels = json.load(f)

        with open(basename + '.vocab', 'r') as f:
            model.vocab = json.load(f)
        return model


class ConvModel(GraphWordClassifierBase):

    def __init__(self):
        super(ConvModel, self).__init__()

    def _stacked(self, pooled, **kwargs):
        pdrop = kwargs.get('dropout', 0.5)
        hszs = listify(kwargs.get('hsz', []))
        activation_type = kwargs.get('activation', 'relu')

        if len(hszs) == 0:
            return pooled

        last_layer = pooled
        for i, hsz in enumerate(hszs):
            last_layer = Dense(units=hsz, activation=activation_type)(last_layer)
            last_layer = Dropout(rate=pdrop)(last_layer)
        return last_layer

    def _pool(self, embed, **kwargs):
        filtsz = kwargs['filtsz']
        pdrop = kwargs.get('dropout', 0.5)
        cmotsz = kwargs['cmotsz']
        mots = []
        for i, fsz in enumerate(filtsz):
            conv = Conv1D(cmotsz, fsz, activation='relu')(embed)
            gmp = GlobalMaxPooling1D()(conv)
            mots.append(gmp)

        joined = keras.layers.concatenate(mots, axis=1)
        cmotsz_all = cmotsz * len(filtsz)
        drop1 = Dropout(pdrop)(joined)

        last_layer = drop1
        return last_layer, cmotsz_all


class SequentialWordClassifierBase(WordClassifierBase):

    def __init__(self):
        super(SequentialWordClassifierBase, self).__init__()

    @classmethod
    def load(cls, basename, **kwargs):
        model = cls()

        model.impl = keras.models.load_model(basename, **kwargs)
        with open(basename + '.labels', 'r') as f:
            model.labels = json.load(f)

        with open(basename + '.vocab', 'r') as f:
            model.vocab = json.load(f)
        return model

    def _stacked(self, **kwargs):
        pdrop = kwargs.get('dropout', 0.5)
        hszs = listify(kwargs.get('hsz', []))
        activation_type = kwargs.get('activation', 'relu')

        if len(hszs) == 0:
            return

        for i, hsz in enumerate(hszs):
            self.impl.add(Dense(units=hsz, activation=activation_type))
            self.impl.add(Dropout(rate=pdrop))

    def _pool(self, dsz, **kwargs):
        pass

    def _embed(self, w2v, **kwargs):
        finetune = bool(kwargs.get('finetune', True))
        mxlen = int(kwargs.get('mxlen', 100))
        vocab_size = w2v.weights.shape[0]
        embedding_dim = w2v.dsz
        self.impl.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                weights=[w2v.weights], input_length=mxlen, trainable=finetune))

    @classmethod
    def create(cls, embeddings, labels, **kwargs):
        w2v = embeddings['word']
        model = cls()
        model.labels = labels
        model.vocab = w2v.vocab
        nc = len(labels)
        model.impl = keras.models.Sequential()
        model._embed(w2v)
        model._pool(dsz=w2v.dsz, **kwargs)
        model._stacked(**kwargs)
        model.impl.add(Dense(units=nc, activation='softmax'))
        model.impl.summary()
        return model


class LSTMModel(SequentialWordClassifierBase):

    def __init__(self):
        super(LSTMModel, self).__init__()

    def _embed(self, w2v, **kwargs):
        finetune = bool(kwargs.get('finetune', True))
        mxlen = int(kwargs.get('mxlen', 100))
        vocab_size = w2v.weights.shape[0]
        embedding_dim = w2v.dsz
        self.impl.add(Embedding(input_dim=vocab_size, mask_zero=True, output_dim=embedding_dim,
                                weights=[w2v.weights], input_length=mxlen, trainable=finetune))
    
    def _pool(self, dsz, **kwargs):
        pdrop = kwargs.get('dropout', 0.5)
        mxlen = int(kwargs.get('mxlen', 100))
        nlayers = kwargs.get('layers', 1)
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 100))
        insz = dsz

        if type(hsz) is list:
            hsz = hsz[0]
        for _ in range(nlayers-1):
            self.impl.add(LSTM(hsz, return_sequences=True, input_shape=(mxlen, insz)))
            insz = hsz
        self.impl.add(LSTM(hsz))
        self.impl.add(Dropout(pdrop))

    @staticmethod
    def load(basename, **kwargs):
        model = LSTMModel()

        model.impl = keras.models.load_model(basename, **kwargs)
        with open(basename + '.labels', 'r') as f:
            model.labels = json.load(f)

        with open(basename + '.vocab', 'r') as f:
            model.vocab = json.load(f)
        return model


class NBowModel(SequentialWordClassifierBase):
    def __init__(self):
        super(NBowModel, self).__init__()

    def _pool(self, dsz, **kwargs):
        pdrop = kwargs.get('dropout', 0.5)
        self.impl.add(GlobalAveragePooling1D())
        self.impl.add(Dropout(rate=pdrop))


class NBowMaxModel(NBowModel):

    def __init__(self):
        super(NBowMaxModel, self).__init__()

    def _pool(self, dsz, **kwargs):
        pdrop = kwargs.get('dropout', 0.5)
        self.impl.add(GlobalMaxPooling1D())
        self.impl.add(Dropout(rate=pdrop))

    @staticmethod
    def load(basename, **kwargs):
        model = NBowMaxModel()

        model.impl = keras.models.load_model(basename, **kwargs)
        with open(basename + '.labels', 'r') as f:
            model.labels = json.load(f)

        with open(basename + '.vocab', 'r') as f:
            model.vocab = json.load(f)
        return model


BASELINE_CLASSIFICATION_MODELS = {
    'default': ConvModel.create,
    'lstm': LSTMModel.create,
    'nbowmax': NBowMaxModel.create,
    'nbow': NBowModel.create
}
BASELINE_CLASSIFICATION_LOADERS = {
    'default': ConvModel.load,
    'lstm': LSTMModel.load,
    'nbowmax': NBowMaxModel.create,
    'nbow': NBowModel.create
}


def create_model(embeddings, labels, **kwargs):
    return create_classifier_model(BASELINE_CLASSIFICATION_MODELS, embeddings, labels, **kwargs)


def load_model(outname, **kwargs):
    return load_classifier_model(BASELINE_CLASSIFICATION_LOADERS, outname, **kwargs)

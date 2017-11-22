import keras.models
from keras.layers import Dense, Conv1D, Embedding, Input, merge, GlobalMaxPooling1D, Dropout
from keras.utils import np_utils

from baseline.model import Classifier, load_classifier_model, create_classifier_model
import json


class ConvModel(Classifier):

    def __init__(self):
        pass

    def save(self, basename):
        self.impl.save(basename, overwrite=True)

        with open(basename + '.labels', 'w') as f:
            json.dump(self.labels, f)

        with open(basename + '.vocab', 'w') as f:
            json.dump(self.vocab, f)

    @staticmethod
    def load(basename, **kwargs):
        model = ConvModel()

        model.impl = keras.models.load_model(basename, **kwargs)
        with open(basename + '.labels', 'r') as f:
            model.labels = json.load(f)

        with open(basename + '.vocab', 'r') as f:
            model.vocab = json.load(f)
        return model

    def classify(self, batch_time):
        batchsz = batch_time.shape[0]
        return self.impl.fit(batch_time, batch_size=batchsz)

    @staticmethod
    def create(w2v, labels, **kwargs):
        model = ConvModel()
        model.labels = labels
        model.vocab = w2v.vocab
        filtsz = kwargs['filtsz']
        pdrop = kwargs.get('dropout', 0.5)
        mxlen = int(kwargs.get('mxlen', 100))
        cmotsz = kwargs['cmotsz']
        finetune = bool(kwargs.get('finetune', True))
        nc = len(labels)
        x = Input(shape=(mxlen,), dtype='int32', name='input')

        vocab_size = w2v.weights.shape[0]
        embedding_dim = w2v.dsz

        lut = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[w2v.weights], input_length=mxlen, trainable=finetune)

        embed = lut(x)

        mots = []
        for i, fsz in enumerate(filtsz):
            conv = Conv1D(cmotsz, fsz, activation='relu')(embed)
            gmp = GlobalMaxPooling1D()(conv)
            mots.append(gmp)

        joined = merge(mots, mode='concat')
        cmotsz_all = cmotsz * len(filtsz)
        drop1 = Dropout(pdrop)(joined)

        input_dim = cmotsz_all
        last_layer = drop1
        dense = Dense(output_dim=nc, input_dim=input_dim, activation='softmax')(last_layer)
        model.impl = keras.models.Model(input=[x], output=[dense])
        return model

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

# TODO: Add the other models!
BASELINE_CLASSIFICATION_MODELS = {
    'default': ConvModel.create
}
BASELINE_CLASSIFICATION_LOADERS = {
    'default': ConvModel.load
}


def create_model(w2v, labels, **kwargs):
    return create_classifier_model(BASELINE_CLASSIFICATION_MODELS, w2v, labels, **kwargs)


def load_model(outname, **kwargs):
    return load_classifier_model(BASELINE_CLASSIFICATION_LOADERS, outname, **kwargs)

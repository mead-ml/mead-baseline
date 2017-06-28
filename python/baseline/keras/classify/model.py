from keras.models import Model, load_model
from keras.layers import Dense, Convolution1D, Embedding, Input, merge, GlobalMaxPooling1D, Dropout
import json

class ConvModelKeras:

    def __init__(self):
        pass

    def save(self, basename):
        self.impl.save(basename, overwrite=True)

        with open(basename + '.labels', 'w') as f:
            json.dump(self.labels, f)

        with open(basename + '.vocab', 'w') as f:
            json.dump(self.vocab, f)

    def load(self, basename):
        self.impl = load_model(basename)
        with open(basename + '.labels', 'r') as f:
            self.labels = json.load(f)

        with open(basename + '.vocab', 'r') as f:
            self.vocab = json.load(f)


    @staticmethod
    def create(w2v, labels, **kwargs):
        model = ConvModelKeras()
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
            conv = Convolution1D(cmotsz, fsz, activation='relu', input_length=mxlen)(embed)
            gmp = GlobalMaxPooling1D()(conv)
            mots.append(gmp)

        joined = merge(mots, mode='concat')
        cmotsz_all = cmotsz * len(filtsz)
        drop1 = Dropout(pdrop)(joined)

        input_dim = cmotsz_all
        last_layer = drop1
        dense = Dense(output_dim=nc, input_dim=input_dim, activation='softmax')(last_layer)
        model.impl = Model(input=[x], output=[dense])
        return model


# Use the functional API since we support parallel convolutions
def create_model(w2v, labels, **kwargs):
    return ConvModelKeras.create(w2v, labels, **kwargs)
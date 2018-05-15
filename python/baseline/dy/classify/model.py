from itertools import chain
import dynet as dy
from baseline.model import (
    Classifier,
    load_classifier_model,
    create_classifier_model
)
from baseline.dy.dynety import *


class WordClassifierBase(Classifier):

    def __init__(
            self,
            embeddings, labels,
            finetune=True, dense=False,
            dropout=0.5, batched=False,
            **kwargs
    ):
        super(WordClassifierBase, self).__init__()
        self._pc = dy.ParameterCollection()

        self.batched = batched
        self.pdrop = dropout
        self.train = True

        vsz = len(embeddings.vocab)
        dsz = embeddings.dsz
        self.vocab = embeddings.vocab

        self.embed = Embedding(
            vsz, dsz, self.pc,
            embeddings.weights, finetune, dense, self.batched
        )

        self.labels = labels
        n_classes = len(self.labels)

        pool_size, self.pool = self._init_pool(dsz, **kwargs)
        stack_size, self.stacked = self._init_stacked(pool_size, **kwargs)
        self.output = self._init_output(stack_size, n_classes)

    @property
    def pc(self):
        return self._pc

    def __str__(self):
        str_ = []
        for p in chain(self.pc.lookup_parameters_list(), self.pc.parameters_list()):
            str_.append("{}: {}".format(p.name(), p.shape()))
        str_ = '\n'.join(str_)
        if self.batched:
            return "Batched Model: \n{}".format(str_)
        return str_

    def make_input(self, batch_dict):
        x = batch_dict['x']
        y = batch_dict['y']
        if self.batched:
            return x.T, y.T
        return x[0], y[0]

    def forward(self, input_):
        embedded = self.embed(input_)
        pooled = self.pool(embedded)
        stacked = self.stacked(pooled)
        return self.output(stacked)

    def loss(self, input_, y):
        if self.batched:
            return dy.pickneglogsoftmax_batch(input_, y)
        return dy.pickneglogsoftmax(input_, y)

    def dropout(self, input_):
        if self.train:
            return dy.dropout(input_, self.pdrop)
        return input_

    def _init_stacked(self, input_dim, hsz=None, layers=1, **kwargs):
        if hsz is None:
            hsz = kwargs.get("cmotsz", 100)
        stacked_layers = []
        isz = input_dim
        for x in range(layers):
            stacked_layers.append(Linear(hsz, isz, self.pc))
            stacked_layers.append(dy.rectify)
            stacked_layers.append(self.dropout)
            isz = hsz

        def call_stacked(input_):
            for layer in stacked_layers:
                input_ = layer(input_)
            return input_

        return hsz, call_stacked

    def _init_output(self, input_dim, n_classes):
        return Linear(n_classes, input_dim, self.pc, name="Output")

    @classmethod
    def create(cls, embeddings_set, labels, **kwargs):
        embeddings = embeddings_set['word']
        model = cls(embeddings, labels, **kwargs)
        print(model)
        return model

    def save(self, file_name):
        self.pc.save(file_name)
        return self

    def load(self, file_name):
        self.pc.populate(file_name)
        return self

class ConvModel(WordClassifierBase):
    def __init__(self, *args, **kwargs):
        kwargs['dense'] = True
        super(ConvModel, self).__init__(*args, **kwargs)

    def _init_pool(self, dsz, filtsz, cmotsz, **kwargs):
        convs = []
        for fsz in filtsz:
            convs.append(Convolution1d(fsz, cmotsz, dsz, self.pc))

        def call_pool(input_):
            input_ = dy.reshape(input_, (1, *input_.dim()[0]))
            mots = []
            for conv in convs:
                mots.append(conv(input_))
            return dy.concatenate(mots)

        return len(filtsz) * cmotsz, call_pool

class LSTMModel(WordClassifierBase):

    def _init_pool(self, dsz, hsz=None, layers=1, **kwargs):
        if hsz is None:
            hsz = kwargs.get('cmotsz', 100)
        return hsz, LSTM(hsz, dsz, self.pc, layers=layers)

class NBowModel(WordClassifierBase):

    def _init_pool(self, *args, **kwargs):
        def pool(input_):
            return dy.esum(input_) / len(input_)

        return args[0], pool

class NBowMax(WordClassifierBase):

    def _init_pool(self, *args, **kwargs):
        def pool(input_):
            return dy.emax(input_)

        return args[0], pool


BASELINE_CLASSIFICATION_MODELS = {
    'default': ConvModel.create,
    'lstm': LSTMModel.create,
    'nbow': NBowModel.create,
    'nbowmax': NBowModel.create,
}

BASELINE_CLASSIFICATION_LOADER = {
    'default': ConvModel.load,
    'lstm': LSTMModel.load,
    'nbow': NBowModel.load,
    'nbowmax': NBowMax.load
}

def create_model(embeddings, labels, **kwargs):
    return create_classifier_model(BASELINE_CLASSIFICATION_MODELS, embeddings, labels, **kwargs)

def load_model(outname, **kwargs):
    return load_classifier_model(BASELINE_CLASSIFICATION_LOADERS, outname, **kwargs)

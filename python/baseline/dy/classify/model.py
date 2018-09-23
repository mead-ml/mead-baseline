import dynet as dy
from baseline.model import (
    Classifier,
    load_classifier_model,
    create_classifier_model
)
from baseline.utils import listify
from baseline.dy.dynety import *


class WordClassifierModelBase(Classifier, DynetModel):

    def __init__(
            self,
            embeddings, labels,
            finetune=True, dense=False,
            dropout=0.5, batched=False,
            **kwargs
    ):
        super(WordClassifierBase, self).__init__()

        self.batched = batched
        self.pdrop = dropout
        self.train = True

        vsz = len(embeddings['word'].vocab)
        dsz = embeddings['word'].dsz

        self.vocab = {}
        self.vocab['word'] = embeddings['word'].vocab

        self.embed_word = Embedding(
            vsz, dsz, self.pc,
            embeddings['word'].weights, finetune, dense, self.batched
        )

        self.labels = labels
        n_classes = len(self.labels)

        pool_size, self.pool = self._init_pool(dsz, **kwargs)
        stack_size, self.stacked = self._init_stacked(pool_size, **kwargs)
        self.output = self._init_output(stack_size, n_classes)

    def __str__(self):
        str_ = super(WordClassifierBase, self).__str__()
        if self.batched:
            return "Batched Model: \n{}".format(str_)
        return str_

    def make_input(self, batch_dict):
        x = batch_dict['x']
        y = batch_dict['y']
        lengths = batch_dict['lengths']
        if self.batched:
            return x.T, y.T, lengths.T
        return x[0], y[0], lengths

    def forward(self, input_, lengths):
        embedded = self.embed_word(input_)
        pooled = self.pool(embedded, lengths)
        stacked = pooled if self.stacked is None else self.stacked(pooled)
        return self.output(stacked)

    def loss(self, input_, y):
        if self.batched:
            return dy.pickneglogsoftmax_batch(input_, y)
        return dy.pickneglogsoftmax(input_, y)

    def dropout(self, input_):
        if self.train:
            return dy.dropout(input_, self.pdrop)
        return input_

    def _init_stacked(self, input_dim, **kwargs):

        hszs = listify(kwargs.get('hsz', []))
        if len(hszs) == 0:
            return input_dim, None

        stacked_layers = []
        isz = input_dim
        for i, hsz in enumerate(hszs):
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
        model = cls(embeddings_set, labels, **kwargs)
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
        parallel_conv = ParallelConv(filtsz, cmotsz, dsz, self.pc)
        def call_pool(input_, _):
            return parallel_conv(input_)

        return len(filtsz) * cmotsz, call_pool


class LSTMModel(WordClassifierBase):

    def _init_pool(self, dsz, layers=1, **kwargs):
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 100))
        if type(hsz) is list:
            hsz = hsz[0]
        self.rnn = dy.VanillaLSTMBuilder(layers, dsz, hsz, self.pc)

        def pool(input_, lengths):
            return rnn_encode(self.rnn, input_, lengths)

        return hsz, pool


class BLSTMModel(WordClassifierBase):

    def _init_pool(self, dsz, layers=1, **kwargs):
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 100))
        if type(hsz) is list:
            hsz = hsz[0]
        self.rnn = dy.BiRNNBuilder(layers, dsz, hsz, self.pc, dy.VanillaLSTMBuilder)

        def pool(input_, lengths):
            return rnn_encode(self.rnn, input_, lengths)

        return hsz, pool


class NBowModel(WordClassifierBase):

    def _init_pool(self, *args, **kwargs):
        def pool(input_, _):
            return dy.esum(input_) / len(input_)

        return args[0], pool


class NBowMax(WordClassifierBase):

    def _init_pool(self, *args, **kwargs):
        def pool(input_, _):
            return dy.emax(input_)

        return args[0], pool


BASELINE_CLASSIFICATION_MODELS = {
    'default': ConvModel.create,
    'lstm': LSTMModel.create,
    'blstm': BLSTMModel.create,
    'nbow': NBowModel.create,
    'nbowmax': NBowMax.create,
}

BASELINE_CLASSIFICATION_LOADER = {
    'default': ConvModel.load,
    'lstm': LSTMModel.load,
    'blstm': BLSTMModel.load,
    'nbow': NBowModel.load,
    'nbowmax': NBowMax.load
}


def create_model(embeddings, labels, **kwargs):
    return create_classifier_model(BASELINE_CLASSIFICATION_MODELS, embeddings, labels, **kwargs)


def load_model(outname, **kwargs):
    return load_classifier_model(BASELINE_CLASSIFICATION_LOADER, outname, **kwargs)

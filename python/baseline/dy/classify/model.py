import dynet as dy
from baseline.model import (
    ClassifierModel,
    load_classifier_model,
    create_classifier_model
)
from baseline.utils import listify
from baseline.dy.dynety import *


class ClassifierModelBase(DynetModel, ClassifierModel):

    def __init__(self, embeddings, labels, dropout=0.5, batched=False, **kwargs):
        super(ClassifierModelBase, self).__init__(kwargs['pc'])

        self.batched = batched
        self.pdrop = dropout
        self.train = True
        self.labels = labels
        n_classes = len(self.labels)
        dsz = self._init_embed(embeddings)
        pool_size, self.pool = self._init_pool(dsz, **kwargs)
        stack_size, self.stacked = self._init_stacked(pool_size, **kwargs)
        self.output = self._init_output(stack_size, n_classes)
        self.lengths_key = kwargs.get('lengths_key')

    def __str__(self):
        str_ = super(ClassifierModelBase, self).__str__()
        if self.batched:
            return "Batched Model: \n{}".format(str_)
        return str_

    def _init_embed(self, embeddings):
        dsz = 0
        self.embeddings = embeddings
        for embedding in self.embeddings.values():
            dsz += embedding.get_dsz()
        return dsz

    def _embed(self, batch_dict):
        all_embeddings_lists = []
        for k, embedding in self.embeddings.items():
            all_embeddings_lists += [embedding.encode(batch_dict[k])]

        embed = dy.concatenate(all_embeddings_lists, d=1)
        return embed


    def make_input(self, batch_dict):
        example_dict = dict({})
        for k in self.embeddings.keys():
            if self.batched:
                example_dict[k] = batch_dict[k].T
            else:
                example_dict[k] = batch_dict[k][0]

        if self.lengths_key is not None:
            lengths = batch_dict[self.lengths_key]
            if self.batched:
                example_dict['lengths'] = lengths.T
            else:
                example_dict['lengths'] = lengths

        if 'y' in batch_dict:
            example_dict['y'] = batch_dict['y']
        return example_dict

    def forward(self, batch_dict):

        embedded = self._embed(batch_dict)
        pooled = self.pool(embedded, batch_dict['lengths'])
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


class ConvModel(ClassifierModelBase):
    def __init__(self, *args, **kwargs):
        super(ConvModel, self).__init__(*args, **kwargs)

    def _init_pool(self, dsz, filtsz, cmotsz, **kwargs):
        parallel_conv = ParallelConv(filtsz, cmotsz, dsz, self.pc)
        def call_pool(embedded, _):
            conv = self.dropout(parallel_conv(embedded))
            return conv

        return len(filtsz) * cmotsz, call_pool


class LSTMModel(ClassifierModelBase):

    def _init_pool(self, dsz, layers=1, **kwargs):
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 100))
        if type(hsz) is list:
            hsz = hsz[0]
        self.rnn = dy.VanillaLSTMBuilder(layers, dsz, hsz, self.pc)

        def pool(input_, lengths):
            return rnn_encode(self.rnn, input_, lengths)

        return hsz, pool


class BLSTMModel(ClassifierModelBase):

    def _init_pool(self, dsz, layers=1, **kwargs):
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 100))
        if type(hsz) is list:
            hsz = hsz[0]
        self.rnn = dy.BiRNNBuilder(layers, dsz, hsz, self.pc, dy.VanillaLSTMBuilder)

        def pool(input_, lengths):
            return rnn_encode(self.rnn, input_, lengths)

        return hsz, pool


class NBowModel(ClassifierModelBase):

    def _init_pool(self, *args, **kwargs):
        def pool(input_, _):
            return dy.esum(input_) / len(input_)

        return args[0], pool


class NBowMax(ClassifierModelBase):

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

import logging
import dynet as dy
from baseline.model import (
    ClassifierModel, register_model
)
from baseline.utils import listify
from baseline.dy.dynety import *

logger = logging.getLogger('baseline')


class ClassifierModelBase(DynetModel, ClassifierModel):

    def __init__(self, embeddings, labels, dropout=0.5, batched=False, **kwargs):
        super(ClassifierModelBase, self).__init__(kwargs['pc'])

        self.batched = batched
        self.pdrop = dropout
        self.train = True
        self.labels = labels
        n_classes = len(self.labels)
        dsz = self.init_embed(embeddings)
        pool_size, self.pool = self.init_pool(dsz, **kwargs)
        stack_size, self.stacked = self.init_stacked(pool_size, **kwargs)
        self.output = self.init_output(stack_size, n_classes)
        self.lengths_key = kwargs.get('lengths_key')

    def __str__(self):
        str_ = super(ClassifierModelBase, self).__str__()
        if self.batched:
            return "Batched Model: \n{}".format(str_)
        return str_

    def init_embed(self, embeddings):
        dsz = 0
        self.embeddings = embeddings
        for embedding in self.embeddings.values():
            dsz += embedding.get_dsz()
        return dsz

    def embed(self, batch_dict):
        all_embeddings_lists = []
        for k, embedding in self.embeddings.items():
            all_embeddings_lists.append(embedding.encode(batch_dict[k]))

        embedded = dy.concatenate(all_embeddings_lists, d=1)
        return embedded

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
            if self.batched:
                example_dict['y'] = batch_dict['y'].T
            else:
                example_dict['y'] = batch_dict['y'][0]
        return example_dict

    def forward(self, batch_dict):

        embedded = self.embed(batch_dict)
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

    def init_stacked(self, input_dim, **kwargs):

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

    def init_pool(self, dsz, **kwargs):
        pass

    def init_output(self, input_dim, n_classes):
        return Linear(n_classes, input_dim, self.pc, name="Output")

    @classmethod
    def create(cls, embeddings_set, labels, **kwargs):
        model = cls(embeddings_set, labels, **kwargs)
        logger.info(model)
        return model

    def save(self, file_name):
        self.pc.save(file_name)
        return self

    def load(self, file_name):
        self.pc.populate(file_name)
        return self


@register_model(task='classify', name='default')
class ConvModel(ClassifierModelBase):
    def __init__(self, *args, **kwargs):
        super(ConvModel, self).__init__(*args, **kwargs)

    def init_pool(self, dsz, **kwargs):
        filtsz = kwargs['filtsz']
        cmotsz = kwargs['cmotsz']

        parallel_conv = ParallelConv(filtsz, cmotsz, dsz, self.pc)
        def call_pool(embedded, _):
            conv = self.dropout(parallel_conv(embedded))
            return conv

        return len(filtsz) * cmotsz, call_pool


@register_model(task='classify', name='lstm')
class LSTMModel(ClassifierModelBase):

    def init_pool(self, dsz, **kwargs):
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 100))
        rnntype = kwargs.get('rnn_type', kwargs.get('rnntype', 'lstm'))

        layers = kwargs.get('layers', 1)
        if type(hsz) is list:
            hsz = hsz[0]

        if rnntype.startswith('b'):
            self.rnn = dy.BiRNNBuilder(layers, dsz, hsz, self.pc, dy.VanillaLSTMBuilder)
        else:
            self.rnn = dy.VanillaLSTMBuilder(layers, dsz, hsz, self.pc)

        def pool(input_, lengths):
            return rnn_encode(self.rnn, input_, lengths)

        return hsz, pool


@register_model(task='classify', name='nbow')
class NBowModel(ClassifierModelBase):

    def init_pool(self, dsz, **kwargs):
        def pool(input_, _):
            return dy.esum(input_) / len(input_)

        return dsz, pool


@register_model(task='classify', name='nbowmax')
class NBowMax(ClassifierModelBase):

    def init_pool(self, dsz, **kwargs):
        def pool(input_, _):
            return dy.emax(input_)

        return dsz, pool

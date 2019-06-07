import keras.models
from keras.layers import (Dense,
                          Conv1D,
                          GlobalMaxPooling1D,
                          Dropout,
                          LSTM,
                          GlobalAveragePooling1D)

from keras.utils import np_utils
from keras import backend as K
from baseline.keras.embeddings import LookupTableEmbeddings
from baseline.version import __version__
from baseline.utils import listify, ls_props, write_json, read_json
from baseline.model import ClassifierModel, register_model 
#load_classifier_model, create_classifier_model
import json


class ClassifierModelBase(ClassifierModel):

    def __init__(self):
        pass

    def save(self, basename):
        self.impl.save(basename, overwrite=True)
        path = basename.split('/')
        base = path[-1]
        outdir = '/'.join(path[:-1])

        # For each embedding, save a record of the keys

        embeddings_info = {}
        for k, v in self.embeddings.items():
            embeddings_info[k] = v.__class__.__name__
        state = {
            "version": __version__,
            "embeddings": embeddings_info
            ## "lengths_key": self.lengths_key
        }
        for prop in ls_props(self):
            state[prop] = getattr(self, prop)

        write_json(state, basename + '.state')
        write_json(self.labels, basename + ".labels")
        for key, embedding in self.embeddings.items():
            embedding.save_md(basename + '-{}-md.json'.format(key))

    #def classify(self, batch_dict):
    #    batch_time = batch_dict['x']
    #    batchsz = batch_time.shape[0]
    #    return self.impl.fit(batch_time, batch_size=batchsz)

    def predict(self, batch_dict):
        example_dict = self.make_input(batch_dict)
        probs = self.impl.predict_on_batch(example_dict)
        results = []
        batchsz = probs.shape[0]
        for b in range(batchsz):
            outcomes = [(self.labels[id_i], prob_i) for id_i, prob_i in enumerate(probs[b])]
            results.append(outcomes)
        return results

    def make_input(self, batch_dict):
        example_dict = dict({})
        for k, embedding in self.embeddings.items():
            example_dict[k] = batch_dict[k]

        y = batch_dict.get('y')
        if y is not None:
            y = np_utils.to_categorical(batch_dict['y'], len(self.labels))
            example_dict['y'] = y
        return example_dict

    def get_labels(self):
        return self.labels

    def get_vocab(self):
        return self.vocab


class GraphWordClassifierBase(ClassifierModelBase):

    def __init__(self):
        super(GraphWordClassifierBase, self).__init__()

    def _embed(self):
        embeds = []
        dsz = 0
        for k, embedding in self.embeddings.items():
            embeds.append(embedding.encode())
            dsz += embedding.dsz

        return keras.layers.concatenate(embeds) if len(embeds) > 1 else embeds[0], dsz

    @classmethod
    def create(cls, embeddings, labels, **kwargs):
        model = cls()
        model.embeddings = embeddings
        model.labels = labels
        nc = len(labels)
        embed, dsz = model._embed()
        last_layer = model._pool(embed, dsz, **kwargs)
        last_layer = model._stacked(last_layer, **kwargs)
        dense = Dense(units=nc, activation='softmax')(last_layer)
        model.impl = keras.models.Model(inputs=[e.x for e in model.embeddings.values()], outputs=[dense])
        model.impl.summary()
        return model

    def _pool(self, embed, insz, **kwargs):
        pass

    @classmethod
    def load(cls, basename, **kwargs):
        K.clear_session()
        model = cls()
        model.impl = keras.models.load_model(basename)
        state = read_json(basename + '.state')
        for prop in ls_props(model):
            if prop in state:
                setattr(model, prop, state[prop])
        inputs = dict({(v.name[:v.name.find(':')], v) for v in model.impl.inputs})

        model.embeddings = dict()
        for key, class_name in state['embeddings'].items():
            md = read_json('{}-{}-md.json'.format(basename, key))
            embed_args = dict({'vsz': md['vsz'], 'dsz': md['dsz']})
            embed_args[key] = inputs[key]
            Constructor = eval(class_name)
            model.embeddings[key] = Constructor(key, **embed_args)

        ##model.lengths_key = state.get('lengths_key')

        with open(basename + '.labels', 'r') as f:
            model.labels = json.load(f)

        return model

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

@register_model(task='classify', name='default')
class ConvModel(GraphWordClassifierBase):

    def __init__(self):
        super(ConvModel, self).__init__()

    def _pool(self, embed, insz, **kwargs):
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
        return last_layer ##, cmotsz_all

@register_model(task='classify', name='lstm')
class LSTMModel(GraphWordClassifierBase):

    def __init__(self):
        super(LSTMModel, self).__init__()

    def _pool(self, embed, dsz, **kwargs):
        pdrop = kwargs.get('dropout', 0.5)
        mxlen = int(kwargs.get('mxlen', 100))
        nlayers = kwargs.get('layers', 1)
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 100))

        insz = dsz
        last_layer = embed
        if type(hsz) is list:
            hsz = hsz[0]
        for _ in range(nlayers-1):
            last_layer = LSTM(hsz, return_sequences=True, input_shape=(mxlen, insz))(last_layer)
            insz = hsz
        last_layer = LSTM(hsz, return_sequences=False)(last_layer)
        drop1 = Dropout(pdrop)(last_layer)
        last_layer = drop1
        return last_layer ##, hsz

@register_model(task='classify', name='nbow')
class NBoWModel(GraphWordClassifierBase):

    def __init__(self, PoolingLayer=GlobalAveragePooling1D):
        super(NBoWModel, self).__init__()
        self.PoolingLayer = PoolingLayer

    def _pool(self, last_layer, dsz, **kwargs):
        pdrop = kwargs.get('dropout', 0.5)
        last_layer = self.PoolingLayer()(last_layer)
        drop1 = Dropout(pdrop)(last_layer)
        last_layer = drop1
        return last_layer

@register_model(task='classify', name='nbow_max')
class NBoWMaxModel(NBoWModel):

    def __init__(self):
        super(NBoWMaxModel, self).__init__(GlobalMaxPooling1D)

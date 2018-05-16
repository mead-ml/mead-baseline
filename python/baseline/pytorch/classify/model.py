import torch
import torch.nn as nn
import math
import json
from baseline.model import Classifier, load_classifier_model, create_classifier_model
from baseline.pytorch.torchy import *
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


class WordClassifierBase(nn.Module, Classifier):

    def __init__(self):
        super(WordClassifierBase, self).__init__()

    @classmethod
    def load(cls, outname, **kwargs):
        model = torch.load(outname)
        return model

    def save(self, outname):
        print('saving %s' % outname)
        torch.save(self, outname)

    @classmethod
    def create(cls, embeddings_set, labels, **kwargs):
        embeddings = embeddings_set['word']
        finetune = kwargs.get('finetune', True)
        dsz = embeddings.dsz
        model = cls()
        model.pdrop = kwargs.get('dropout', 0.5)
        model.labels = labels
        nc = len(labels)
        model.vocab = embeddings.vocab
        model.lut = pytorch_embedding(embeddings, finetune)
        pool_dim = model._init_pool(dsz, **kwargs)
        stacked_dim = model._init_stacked(pool_dim, **kwargs)
        model._init_output(stacked_dim, nc)
        print(model)
        return model

    def create_loss(self):
        return nn.NLLLoss()

    def __init__(self):
        super(WordClassifierBase, self).__init__()

    def make_input(self, batch_dict):
        x = batch_dict['x']
        y = batch_dict['y']
        if type(x) == list:
            x = [torch.autograd.Variable(item.cuda()) for item in x]
        else:
            x = torch.autograd.Variable(x.cuda())
        y = torch.autograd.Variable(y.cuda())
        return x, y

    def forward(self, input):
        # BxTxC
        embeddings = self.lut(input)
        pooled = self._pool(embeddings)
        stacked = self._stacked(pooled)
        return self.output(stacked)

    def classify(self, batch_dict):
        return classify_bt(self, batch_dict['x'])

    def get_labels(self):
        return self.labels

    def get_vocab(self):
        return self.vocab

    def _pool(self, embeddings):
        pass

    def _stacked(self, pooled):
        return pooled

    def _init_stacked(self, input_dim, **kwargs):
        return input_dim

    def _init_output(self, input_dim, nc):
        self.output = nn.Sequential()
        append2seq(self.output, (
            nn.Linear(input_dim, nc),
            nn.LogSoftmax(dim=1)
        ))

    def _init_pool(self, dsz, **kwargs):
        pass


class ConvModel(WordClassifierBase):

    def __init__(self):
        super(ConvModel, self).__init__()

    def _init_pool(self, dsz, **kwargs):
        filtsz = kwargs['filtsz']
        cmotsz = kwargs['cmotsz']
        self.parallel_conv = ParallelConv(dsz, cmotsz, filtsz, "relu", self.pdrop)
        return self.parallel_conv.outsz

    def _pool(self, btc):
        embeddings = btc.transpose(1, 2).contiguous()
        return self.parallel_conv(embeddings)


class LSTMModel(WordClassifierBase):

    def __init__(self):
        super(LSTMModel, self).__init__()

    def _init_pool(self, dsz, **kwargs):
        unif = kwargs['unif']
        hsz = kwargs.get('hsz', kwargs.get('cmotsz', 100))
        self.lstm = nn.LSTM(dsz, hsz, 1, bias=False, batch_first=True, dropout=self.pdrop)
        for weight in self.lstm.parameters():
            weight.data.uniform_(-unif, unif)
        return hsz

    def _pool(self, embeddings):
        output, hidden = self.lstm(embeddings)
        last_frame = output[:, -1, :].squeeze(1)
        return last_frame


class NBowBase(WordClassifierBase):

    def _init__(self):
        super(NBowBase, self)._init__()

    def _init_pool(self, dsz, **kwargs):
        return dsz

    def _init_stacked(self, input_dim, **kwargs):
        hsz = kwargs.get('hsz', kwargs.get('cmotsz', 100))
        self.stacked = nn.Sequential()
        append2seq(self.stacked, [nn.Dropout(self.pdrop)])
        layers = []
        nlayers = kwargs.get('layers', 1)
        in_layer_sz = input_dim
        for i in range(nlayers):
            layers.append(nn.Linear(in_layer_sz, hsz))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.pdrop))
            in_layer_sz = hsz
        append2seq(self.stacked, layers)
        return hsz

    def _stacked(self, pooled):
        return self.stacked(pooled)


class NBowModel(NBowBase):

    def __init__(self):
        super(NBowModel, self).__init__()

    def _pool(self, embeddings):
        return torch.mean(embeddings, 1, False)


class NBowMaxModel(NBowBase):
    def __init__(self):
        super(NBowMaxModel, self).__init__()

    def _pool(self, embeddings):
        dmax, _ = torch.max(embeddings, 1, False)
        return dmax


# These define the possible models for this backend
BASELINE_CLASSIFICATION_MODELS = {
    'default': ConvModel.create,
    'lstm': LSTMModel.create,
    'nbow': NBowModel.create,
    'nbowmax': NBowMaxModel.create
}
BASELINE_CLASSIFICATION_LOADERS = {
    'default': ConvModel.load,
    'lstm': LSTMModel.load,
    'nbow': NBowModel.load,
    'nbowmax': NBowMaxModel.create
}


def create_model(embeddings, labels, **kwargs):
    return create_classifier_model(BASELINE_CLASSIFICATION_MODELS, embeddings, labels, **kwargs)


def load_model(outname, **kwargs):
    return load_classifier_model(BASELINE_CLASSIFICATION_LOADERS, outname, **kwargs)

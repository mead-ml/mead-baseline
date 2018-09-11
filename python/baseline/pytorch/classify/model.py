import torch
import torch.nn as nn
import math
import json
from baseline.model import Classifier, load_classifier_model, create_classifier_model
from baseline.pytorch.torchy import *
from baseline.utils import listify
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
        word_embeddings = embeddings_set['word']
        char_embeddings = embeddings_set.get('char')
        finetune = kwargs.get('finetune', True)
        activation_type = kwargs.get('activation', 'relu')

        model = cls()
        model.gpu = not bool(kwargs.get('nogpu', False))

        model.word_dsz = word_embeddings.dsz
        model.char_dsz = char_embeddings.dsz if char_embeddings is not None else 0
        model.pdrop = kwargs.get('dropout', 0.5)
        model.labels = labels
        model.lut = pytorch_embedding(word_embeddings, finetune)
        model.vocab = {}
        model.vocab['word'] = word_embeddings.vocab
        model.vdrop = bool(kwargs.get('variational_dropout', False))

        if model.char_dsz > 0:
            model.char_lut = pytorch_embedding(char_embeddings)
            model.vocab['char'] = char_embeddings.vocab
            char_filtsz = kwargs.get('cfiltsz', [3])
            char_hsz = kwargs.get('char_hsz', 30)
            model._init_pool_chars(char_hsz, char_filtsz, activation_type)
            input_sz = model.word_dsz + model.char_comp.outsz
        else:
            input_sz = model.word_dsz

        model.log_softmax = nn.LogSoftmax(dim=1)
        nc = len(labels)

        pool_dim = model._init_pool(input_sz, **kwargs)
        stacked_dim = model._init_stacked(pool_dim, **kwargs)
        model._init_output(stacked_dim, nc)
        print(model)
        return model

    def create_loss(self):
        return nn.NLLLoss()

    def __init__(self):
        super(WordClassifierBase, self).__init__()

    def _init_pool_chars(self, char_hsz, char_filtsz, activation_type):
        self.char_comp = ParallelConv(self.char_dsz, char_hsz, char_filtsz, activation_type, self.pdrop)

    def make_input(self, batch_dict):
        x = batch_dict['x']
        xch = batch_dict.get('xch')
        y = batch_dict.get('y')
        lengths = batch_dict.get('lengths')
        if self.gpu:
            x = x.cuda()
            if xch is not None:
                xch = xch.cuda()
            if y is not None:
                y = y.cuda()

        return x, xch, lengths, y

    def forward(self, input):
        # BxTxC
        x = input[0]
        embeddings = self.lut(x)
        if self.char_dsz > 0:
            xch = input[1]
            B, T, W = xch.shape
            embeddings_char = self._char_encoding(xch.view(-1, W)).view(B, T, self.char_comp.outsz)
            embeddings = torch.cat([embeddings, embeddings_char], 2)
        lengths = input[2]
        pooled = self._pool(embeddings, lengths)
        stacked = self._stacked(pooled)
        return self.output(stacked)

    def classify(self, batch_dict):
        x = batch_dict['x']
        xch = batch_dict.get('xch')
        lengths = batch_dict.get('lengths')
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        if xch is not None and type(xch) == np.ndarray:
            xch = torch.from_numpy(xch)
        if lengths is not None and type(lengths) == np.ndarray:
            lengths = torch.from_numpy(lengths)

        with torch.no_grad():
            if self.gpu:
                x = x.cuda()
                if xch is not None:
                    xch = xch.cuda()
            probs = self((x, xch, lengths)).exp()
            probs.div_(torch.sum(probs))
            results = []
            batchsz = probs.size(0)
            for b in range(batchsz):
                outcomes = [(self.labels[id_i], prob_i) for id_i, prob_i in enumerate(probs[b])]
                results.append(outcomes)
        return results

    def get_labels(self):
        return self.labels

    def get_vocab(self, name='word'):
        return self.vocab.get(name)

    def _pool(self, embeddings, lengths):
        pass

    def _stacked(self, pooled):
        if self.stacked is None:
            return pooled
        return self.stacked(pooled)

    def _init_stacked(self, input_dim, **kwargs):
        hszs = listify(kwargs.get('hsz', []))
        if len(hszs) == 0:
            self.stacked = None
            return input_dim
        self.stacked = nn.Sequential()
        layers = []
        in_layer_sz = input_dim
        for i, hsz in enumerate(hszs):
            layers.append(nn.Linear(in_layer_sz, hsz))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.pdrop))
            in_layer_sz = hsz
        append2seq(self.stacked, layers)
        return in_layer_sz

    def _init_output(self, input_dim, nc):
        self.output = nn.Sequential()
        append2seq(self.output, (
            nn.Linear(input_dim, nc),
            nn.LogSoftmax(dim=1)
        ))

    def _init_pool(self, dsz, **kwargs):
        pass

    def _char_encoding(self, xch):

        # For starters we need to perform embeddings for each character
        # (TxB) x W -> (TxB) x W x D
        char_embeds = self.char_lut(xch)
        # (TxB) x D x W
        char_vecs = char_embeds.transpose(1, 2).contiguous()
        mots = self.char_comp(char_vecs)
        return mots


class ConvModel(WordClassifierBase):

    def __init__(self):
        super(ConvModel, self).__init__()

    def _init_pool(self, dsz, **kwargs):
        filtsz = kwargs['filtsz']
        cmotsz = kwargs['cmotsz']
        self.parallel_conv = ParallelConv(dsz, cmotsz, filtsz, "relu", self.pdrop)
        return self.parallel_conv.outsz

    def _pool(self, btc, lengths):
        embeddings = btc.transpose(1, 2).contiguous()
        return self.parallel_conv(embeddings)


class LSTMModel(WordClassifierBase):

    def __init__(self):
        super(LSTMModel, self).__init__()

    def _init_pool(self, dsz, **kwargs):
        unif = kwargs.get('unif')
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 100))
        if type(hsz) is list:
            hsz = hsz[0]
        self.lstm = nn.LSTM(dsz, hsz, 1, bias=True, dropout=self.pdrop)
        if unif is not None:
            for weight in self.lstm.parameters():
                weight.data.uniform_(-unif, unif)
        return hsz

    def _pool(self, embeddings, lengths):

        embeddings = embeddings.transpose(0, 1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths.tolist())
        output, hidden = self.lstm(packed)
        hidden = hidden[0].view(hidden[0].shape[1:])
        return hidden

    def make_input(self, batch_dict):

        x = batch_dict['x']
        xch = batch_dict.get('xch')
        y = batch_dict.get('y')
        lengths = batch_dict['lengths']
        lengths, perm_idx = lengths.sort(0, descending=True)
        x = x[perm_idx]
        if xch is not None:
            xch = xch[perm_idx]
        if y is not None:
            y = y[perm_idx]
        if self.gpu:
            x = x.cuda()
            if xch is not None:
                xch = xch.cuda()
            if y is not None:
                y = y.cuda()

        if y is not None:
            y = y.contiguous()

        return x, xch, lengths, y


class NBowBase(WordClassifierBase):

    def _init__(self):
        super(NBowBase, self)._init__()

    def _init_pool(self, dsz, **kwargs):
        return dsz

    def _init_stacked(self, input_dim, **kwargs):
        kwargs['hsz'] = kwargs.get('hsz', [100])
        return super(NBowBase, self)._init_stacked(input_dim, **kwargs)


class NBowModel(NBowBase):

    def __init__(self):
        super(NBowModel, self).__init__()

    def _pool(self, embeddings):
        return torch.mean(embeddings, 1, False)


class NBowMaxModel(NBowBase):
    def __init__(self):
        super(NBowMaxModel, self).__init__()

    def _pool(self, embeddings, lengths):
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
